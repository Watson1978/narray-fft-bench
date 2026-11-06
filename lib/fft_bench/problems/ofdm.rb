# frozen_string_literal: true

module FFTBench
  module Problems
    # F. OFDM over a multipath channel, a batch of symbols at once along the
    # last axis. QPSK bits go out through ifft and a cyclic prefix, through
    # fftconvolve with the channel, and back through fft and a one-tap
    # equalizer. Without noise every bit must come back, under each norm.
    # With noise the bit errors must sit within 3 sigma of the QPSK error
    # rate averaged over the subcarriers, each at its own |H_k|^2 Eb/N0.
    class Ofdm < Problem
      register :ofdm

      NORMS = %w[backward ortho forward].freeze
      VARIANTS = %w[batch single].freeze
      SIGMAS = 3.0
      PARSEVAL_TOL = { 'fp64' => 5.6e-14, 'fp32' => 6.1e-7 }.freeze

      def run(fixture)
        cp = fixture.attr(:cp)
        bits = fixture.device(:bits)
        symbols = qpsk(bits)
        n = symbols.shape[1]
        h = fixture.device(:channel)
        response = XFT.fft(h, n: n)
        clean = NORMS.to_h do |norm|
          tx = XFT.ifft(symbols, norm: norm)
          equalized = demodulate(channel(prefixed(tx, cp), h), cp, norm, response)
          [norm, { errors: Gates.mismatches(decide(equalized), bits), parseval: Gates.parseval(tx, symbols, n: n, norm: norm),
                   equalized: equalized }]
        end
        gains = FFTBench.host(abs2(response)).to_a
        noisy = noisy_errors(fixture, symbols, bits, h, response, gains, cp)
        { 'clean' => clean, 'noisy' => noisy, 'shift' => shift_checks(clean['ortho'][:equalized], n),
          'values' => values(clean, noisy, gains) }
      end

      def gate(result)
        gates = result['clean'].flat_map do |norm, c|
          [Gates.eq("#{norm}.bit_errors", c[:errors], 0), Gates.le("#{norm}.parseval", c[:parseval], PARSEVAL_TOL.fetch(FFTBench.dtype_name))]
        end
        gates += result['noisy'].map { |db, x| Gates.le("awgn.#{db}dB.sigmas", x[:sigmas], SIGMAS) }
        gates + result['shift'].map { |name, value| Gates.eq("fftshift.#{name}", value, 0) }
      end

      def bench_step(fixture, variant = nil)
        name = variant || VARIANTS.first
        raise Error, "ofdm has no variant #{name}; known: #{VARIANTS.join(', ')}" unless VARIANTS.include?(name)

        cp = fixture.attr(:cp)
        symbols = qpsk(fixture.device(:bits))
        response = XFT.fft(fixture.device(:channel), n: symbols.shape[1])
        link = ->(x) { demodulate(prefixed(XFT.ifft(x, norm: 'ortho'), cp), cp, 'ortho', response) }
        return -> { link.call(symbols) } if name == 'batch'

        rows = Array.new(symbols.shape[0]) { |i| symbols[i, true] }
        -> { rows.each { |x| link.call(x) } }
      end

      private

      def abs2(z) = (z.real * z.real) + (z.imag * z.imag)

      # Bits 2k and 2k + 1 of a symbol set the signs of subcarrier k.
      def qpsk(bits)
        count, width = bits.shape
        pairs = bits.reshape(count, width / 2, 2)
        re = 1 - (2 * pairs[true, true, 0])
        im = 1 - (2 * pairs[true, true, 1])
        (XC.cast(re) + (XC.cast(im) * Complex::I)) / Math.sqrt(2)
      end

      # x / |x| is exactly +1 or -1, so the decided bits are exactly 0 or 1.
      def decide(z)
        count, n = z.shape
        out = XF.zeros(count, n, 2)
        out[true, true, 0] = (1 - (z.real / z.real.abs)) * 0.5
        out[true, true, 1] = (1 - (z.imag / z.imag.abs)) * 0.5
        out.reshape(count, 2 * n)
      end

      def prefixed(x, cp)
        n = x.shape[-1]
        out = x.class.zeros(*x.shape[0...-1], n + cp)
        out[false, 0...cp] = x[false, (n - cp)...n]
        out[false, cp...(n + cp)] = x
        out
      end

      # Each symbol through the channel on its own; the prefix absorbs the
      # transient at its start, and the tail past the symbol is dropped.
      def channel(x, h)
        XFT.fftconvolve(x, h.reshape(1, h.size))[true, 0...x.shape[1]]
      end

      def demodulate(y, cp, norm, response)
        XFT.fft(y[false, cp..-1], norm: norm) / response
      end

      # Unit-energy QPSK symbols, so Eb = 1 / 2 and the complex noise of
      # variance 1 / (2 Eb/N0) per sample keeps that variance per subcarrier
      # under the ortho norm.
      def noisy_errors(fixture, symbols, bits, h, response, gains, cp)
        return {} unless fixture.arrays.key?('noise')

        noise = fixture.device(:noise)
        received = channel(prefixed(XFT.ifft(symbols, norm: 'ortho'), cp), h)
        per_subcarrier = 2 * symbols.shape[0]
        fixture.attr(:spec)['ebn0_db'].to_h do |db|
          ebn0 = 10**(db / 10.0)
          equalized = demodulate(received + (noise * Math.sqrt(1 / (2 * ebn0))), cp, 'ortho', response)
          errors = Gates.mismatches(decide(equalized), bits)
          p = gains.map { |g| 0.5 * Math.erfc(Math.sqrt(g * ebn0)) }
          expected = per_subcarrier * p.sum
          spread = Math.sqrt(per_subcarrier * p.sum { |q| q * (1 - q) })
          [db, { errors: errors, expected: expected, sigmas: (errors - expected).abs / spread }]
        end
      end

      def shift_checks(equalized, n)
        frequencies = XFT.fftshift(XFT.fftfreq(n))
        ascending = (XM::DFloat.new(n).seq - (n / 2)) / n
        constellation = XFT.fftshift(equalized, axes: [-1])
        { 'frequencies' => Gates.mismatches(frequencies, ascending),
          'dc_center' => Gates.mismatches(constellation[true, n / 2], equalized[true, 0]) }
      end

      def values(clean, noisy, gains)
        out = { 'clean_errors' => clean.values.sum { |c| c[:errors] }, 'channel_gain' => gains.sum / gains.size }
        noisy.each { |db, x| out["awgn.#{db}dB.errors"] = x[:errors] }
        out
      end
    end
  end
end
