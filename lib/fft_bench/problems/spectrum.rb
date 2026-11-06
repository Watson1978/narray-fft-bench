# frozen_string_literal: true

module FFTBench
  module Problems
    # G. Correlation and spectral estimates of a sine in white noise, at a
    # power-of-two, a mixed-radix and a prime length. The autocorrelation by
    # Wiener-Khinchin must match the direct sum, fftconvolve and the complex
    # transforms. hfft of the circular autocorrelation must give |rfft(x)|^2.
    # Welch's one-sided density (Hann, 50% overlap) must put the peak on f0,
    # at the height of the sine, over a floor of 2 sigma^2 / fs.
    class Spectrum < Problem
      register :spectrum

      KINDS = %w[pow2 mixed prime].freeze
      SEGMENTS = [256, 1024, 4096].freeze
      CYCLE = [128, 256, 512, 1024, 2048, 4096].product(KINDS).freeze
      VARIANTS = (KINDS.flat_map { |kind| ["wk_#{kind}", *SEGMENTS.map { |l| "welch#{l}_#{kind}" }] } << 'welch_cycle').freeze
      GATE_SEGMENT = 256
      TOL = { 'fp64' => 1.5e-14, 'fp32' => 6e-6 }.freeze
      STATISTICAL_TOL = 0.1
      GUARD = 4

      def run(fixture)
        spec = fixture.attr(:spec)
        transforms = KINDS.to_h do |kind|
          short = fixture.device("short_#{kind}")
          long = fixture.device(kind)
          [kind, { 'short' => transforms(short).merge(direct: fixture.device("short_#{kind}_direct")),
                   'long' => transforms(long), 'welch' => welch_stats(long, spec) }]
        end
        { 'transforms' => transforms, 'spec' => spec, 'values' => values(transforms) }
      end

      def gate(result)
        spec = result['spec']
        tol = TOL.fetch(FFTBench.dtype_name)
        result['transforms'].flat_map do |kind, t|
          short = t['short']
          checks = [Gates.le("#{kind}.short.direct", Gates.rel_max(short[:r], short[:direct]), tol)]
          checks += %w[short long].flat_map do |length|
            x = t[length]
            [Gates.le("#{kind}.#{length}.fftconvolve", Gates.rel_max(x[:convolved], x[:r]), tol),
             Gates.le("#{kind}.#{length}.complex", Gates.rel_max(x[:complex], x[:r]), tol),
             Gates.le("#{kind}.#{length}.hfft", Gates.rel_max(x[:hfft], x[:power]), tol),
             Gates.le("#{kind}.#{length}.ihfft", Gates.rel_max(x[:ihfft], x[:half]), tol)]
          end
          w = t['welch']
          checks + [Gates.eq("#{kind}.welch.peak_frequency", w[:peak_frequency], spec['f0']),
                    Gates.le("#{kind}.welch.peak_height", ((w[:peak_height] / expected_peak(spec, GATE_SEGMENT)) - 1).abs, STATISTICAL_TOL),
                    Gates.le("#{kind}.welch.noise_floor", ((w[:noise_floor] / floor(spec)) - 1).abs, STATISTICAL_TOL)]
        end
      end

      def bench_step(fixture, variant = nil)
        name = variant || VARIANTS.first
        raise Error, "spectrum has no variant #{name}; known: #{VARIANTS.join(', ')}" unless VARIANTS.include?(name)

        fs = fixture.attr(:spec)['fs']
        signals = KINDS.to_h { |kind| [kind, fixture.device(kind)] }
        return -> { CYCLE.each { |length, kind| welch(signals[kind], length, fs) } } if name == 'welch_cycle'

        op, kind = name.split('_')
        x = signals[kind]
        return -> { autocorrelation(x) } if op == 'wk'

        length = Integer(op.delete_prefix('welch'))
        -> { welch(x, length, fs) }
      end

      private

      def abs2(z) = (z.real * z.real) + (z.imag * z.imag)

      # Wiener-Khinchin: zero-padded to 2n, so the correlation does not wrap.
      def autocorrelation(x)
        n = x.size
        XFT.irfft(abs2(XFT.rfft(x, n: 2 * n)), n: 2 * n)[0...n]
      end

      def transforms(x)
        n = x.size
        r = autocorrelation(x)
        circular = r.dup
        circular[1...n] = r[1...n] + r[1...n].reverse
        half = circular[0..(n / 2)]
        spectrum = XFT.hfft(half, n: n)
        { r: r, convolved: XFT.fftconvolve(x, x.reverse)[(n - 1)...((2 * n) - 1)],
          complex: XFT.ifft(abs2(XFT.fft(x, n: 2 * n))).real[0...n],
          hfft: spectrum[0..(n / 2)], power: abs2(XFT.rfft(x)), ihfft: XFT.ihfft(spectrum), half: XC.cast(half) }
      end

      # The segments that start at multiples of L split into two runs, one
      # from 0 and one from L / 2, each a plain reshape. The average does not
      # care about their order.
      def welch(x, length, fs)
        window = hann(length)
        runs = [0, length / 2].map do |start|
          count = (x.size - start) / length
          x[start...(start + (count * length))].reshape(count, length)
        end
        power = runs.map { |segments| abs2(XFT.rfft(segments * window, axis: -1)).sum(axis: 0) }.reduce(:+)
        count = runs.sum { |segments| segments.shape[0] }
        power * one_sided(length) / (fs * (3.0 * length / 8) * count)
      end

      def hann(length)
        0.5 - (0.5 * XM::NMath.cos(XF.new(length).seq * (2 * Math::PI / length)))
      end

      def one_sided(length)
        XF.new((length / 2) + 1).fill(2).tap do |w|
          w[0] = 1
          w[-1] = 1
        end
      end

      def welch_stats(x, spec)
        psd = welch(x, GATE_SEGMENT, spec['fs'])
        peak = FFTBench.scalar(psd.max_index).to_i
        freqs = XFT.rfftfreq(GATE_SEGMENT, d: 1.0 / spec['fs'])
        rest = [psd[1...(peak - GUARD)], psd[(peak + GUARD + 1)...-1]]
        { peak_frequency: FFTBench.scalar(freqs[peak]), peak_height: FFTBench.scalar(psd[peak]),
          noise_floor: rest.sum { |part| FFTBench.scalar(part.sum) } / rest.sum(&:size) }
      end

      # One-sided density of a sine on a bin: A^2 / 2 (sum w)^2 / (fs sum w^2),
      # with sum w = L / 2 and sum w^2 = 3 L / 8 for the periodic Hann window,
      # over the noise floor of the same bin.
      def expected_peak(spec, length)
        (spec['amplitude']**2 / 2 * ((length / 2.0)**2) / (spec['fs'] * 3.0 * length / 8)) + floor(spec)
      end

      def floor(spec) = 2 * (spec['sigma']**2) / spec['fs']

      def values(transforms)
        transforms.each_with_object({}) do |(kind, t), out|
          out["#{kind}.energy"] = FFTBench.scalar(t['long'][:r][0])
          t['welch'].each { |name, value| out["#{kind}.#{name}"] = value }
        end
      end
    end
  end
end
