# frozen_string_literal: true

module FFTBench
  module Problems
    # A. Whisper's log-mel spectrogram: a periodic Hann window over centred,
    # reflect-padded frames (n_fft 400, hop 160), one batched rfft, the power,
    # the [80, 201] mel filters, log10 and Whisper's two clamps. The same
    # spectrogram through the DFT as two matmuls against cosine and sine
    # tables (narray-llm's Whisper::Mel) must agree, and the frames must
    # keep Parseval.
    class WhisperMel < Problem
      register :whisper_mel

      VARIANTS = %w[rfft dft].freeze
      FLOOR = 1.0e-10
      DYNAMIC_RANGE = 8.0
      PATHS_TOL = { 'fp64' => 1e-10, 'fp32' => 1.1e-2 }.freeze
      PARSEVAL_TOL = { 'fp64' => 1.4e-11, 'fp32' => 4.5e-7 }.freeze

      def run(fixture)
        setup = prepare(fixture)
        frames = frame(fixture.device(:waveform), setup)
        spectrum = XFT.rfft(frames, n: setup[:n_fft], axis: -1)
        rfft_mel = log_mel(abs2(spectrum), setup)
        dft_mel = log_mel(dft_power(frames, setup), setup)
        freqs = XFT.rfftfreq(setup[:n_fft], d: 1.0 / setup[:rate])
        spacing = XM::DFloat.new(setup[:bins]).seq * (setup[:rate].to_f / setup[:n_fft])
        { 'paths' => Gates.rel_max(rfft_mel, dft_mel),
          'parseval' => Gates.parseval(frames, spectrum, n: setup[:n_fft], half: setup[:n_fft]),
          'frequencies' => Gates.rel_max(freqs, spacing),
          'values' => { 'frames' => rfft_mel.shape[1], 'mean' => FFTBench.scalar(rfft_mel.mean),
                        'max' => FFTBench.scalar(rfft_mel.max), 'min' => FFTBench.scalar(rfft_mel.min) } }
      end

      def gate(result)
        dtype = FFTBench.dtype_name
        [Gates.le('paths', result['paths'], PATHS_TOL.fetch(dtype)),
         Gates.le('parseval', result['parseval'], PARSEVAL_TOL.fetch(dtype)), Gates.le('frequencies', result['frequencies'], 0)]
      end

      def bench_step(fixture, variant = nil)
        path = variant || VARIANTS.first
        raise Error, "whisper_mel has no variant #{path}; known: #{VARIANTS.join(', ')}" unless VARIANTS.include?(path)

        setup = prepare(fixture)
        waveform = fixture.device(:waveform)
        return -> { log_mel(abs2(XFT.rfft(frame(waveform, setup), n: setup[:n_fft], axis: -1)), setup) } if path == 'rfft'

        -> { log_mel(dft_power(frame(waveform, setup), setup), setup) }
      end

      private

      def abs2(z) = (z.real * z.real) + (z.imag * z.imag)

      def prepare(fixture)
        n_fft = fixture.attr(:n_fft)
        bins = (n_fft / 2) + 1
        n = XF.new(1, n_fft).seq
        angle = XF.new(n_fft, 1).seq * XF.new(1, bins).seq * (-2.0 * Math::PI / n_fft)
        { n_fft: n_fft, hop: fixture.attr(:hop_length), frames: fixture.attr(:nb_max_frames), bins: bins,
          rate: fixture.attr(:sampling_rate), filters: fixture.device(:mel_filters),
          window: 0.5 - (0.5 * XM::NMath.cos(n * (2.0 * Math::PI / n_fft))),
          cos: XM::NMath.cos(angle), sin: XM::NMath.sin(angle) }
      end

      # np.pad(..., mode="reflect"): the edge sample is not repeated.
      def reflect(signal, pad)
        size = signal.size
        out = XF.zeros(size + (2 * pad))
        out[pad...(pad + size)] = signal
        out[0...pad] = signal[1..pad].reverse
        out[(pad + size)...(size + (2 * pad))] = signal[(size - pad - 1)..(size - 2)].reverse
        out
      end

      # Frame t covers padded[t * hop, t * hop + n_fft). With hop 160 and a
      # frame of 400, the frames are three slices of a [blocks, hop] reshape.
      def frame(waveform, s)
        padded = reflect(waveform, s[:n_fft] / 2)
        blocks, rest = s[:n_fft].divmod(s[:hop])
        needed = s[:frames] + blocks + (rest.zero? ? 0 : 1) - 1
        grid = padded[0...(needed * s[:hop])].reshape(needed, s[:hop])
        out = XF.zeros(s[:frames], s[:n_fft])
        blocks.times { |b| out[true, (b * s[:hop])...((b + 1) * s[:hop])] = grid[b...(b + s[:frames]), true] }
        out[true, (blocks * s[:hop])...s[:n_fft]] = grid[blocks...(blocks + s[:frames]), 0...rest] unless rest.zero?
        out * s[:window]
      end

      def dft_power(frames, s)
        real = frames.dot(s[:cos])
        imag = frames.dot(s[:sin])
        (real * real) + (imag * imag)
      end

      # Whisper's clamps: log10 of at least 1e-10, then no more than 8 below
      # the maximum. max(x, m) = (x + m + |x - m|) / 2 keeps the maximum on the
      # device, so a timed step reads nothing back.
      def log_mel(power, s)
        spec = XM::NMath.log10(s[:filters].dot(power.transpose).clip(FLOOR, nil))
        floor = spec.max - DYNAMIC_RANGE
        ((((spec + floor) + (spec - floor).abs) * 0.5) + 4.0) / 4.0
      end
    end
  end
end
