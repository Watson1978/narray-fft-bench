# frozen_string_literal: true

module FFTBench
  module Problems
    # D. Multiplies two big integers through an FFT convolution of their
    # digits in base B. Each path's coefficients are rounded and carried on
    # the host, and must reproduce Ruby's Integer product digit for digit.
    # The fixture's exact coefficients also come from Integer arithmetic, so
    # error is the true distance from them; under 0.5 every digit rounds right.
    class BignumMul < Problem
      register :bignum_mul

      PATHS = %w[fftconvolve rfft_pow2 rfft_exact fft_pow2 fft_exact].freeze

      def run(fixture)
        base = fixture.attr(:base)
        product = integer(fixture[:a], base) * integer(fixture[:b], base)
        expected = digits(product, base, fixture[:a].size + fixture[:b].size)
        a = fixture.device(:a)
        b = fixture.device(:b)
        found = PATHS.to_h { |path| [path, XM::DFloat.cast(coefficients(path, a, b))] }
        carried = found.transform_values { |c| carry(FFTBench.host(c.rint).to_a, base, expected.size) }
        { 'exact' => fixture.device(:exact, XM::DFloat), 'expected' => expected, 'coefficients' => found,
          'carried' => carried, 'values' => values(carried.values.first.first) }
      end

      def gate(result)
        found = result['coefficients']
        per_path = found.flat_map do |path, c|
          digits, overflow = result['carried'][path]
          [Gates.lt("#{path}.error", Gates.max_abs(c - result['exact']), 0.5),
           Gates.eq("#{path}.mismatches", mismatches(digits, overflow, result['expected']), 0)]
        end
        spread = found.values.combination(2).map { |x, y| Gates.max_abs(x - y) }.max
        per_path << Gates.lt('paths.spread', spread, 0.5)
      end

      def bench_step(fixture, variant = nil)
        path = variant || PATHS.first
        raise Error, "bignum_mul has no variant #{path}; known: #{PATHS.join(', ')}" unless PATHS.include?(path)

        a = fixture.device(:a)
        b = fixture.device(:b)
        -> { coefficients(path, a, b).rint }
      end

      private

      def coefficients(path, a, b)
        return XFT.fftconvolve(a, b) if path == 'fftconvolve'

        length = a.size + b.size - 1
        m = path.end_with?('_pow2') ? 1 << (length - 1).bit_length : length
        if path.start_with?('rfft')
          XFT.irfft(XFT.rfft(a, n: m) * XFT.rfft(b, n: m), n: m)[0...length]
        else
          XFT.ifft(XFT.fft(a, n: m) * XFT.fft(b, n: m)).real[0...length]
        end
      end

      def width(base)
        width = Math.log10(base).round
        raise Error, "base #{base} is not a power of 10" unless 10**width == base

        width
      end

      def integer(host_digits, base)
        format = "%0#{width(base)}d"
        host_digits.to_a.reverse.map { |d| format(format, d.to_i) }.join.to_i
      end

      # Little-endian digits of a nonnegative Integer, count of them.
      def digits(integer, base, count)
        text = integer.to_s.rjust(count * width(base), '0')
        raise Error, "#{integer.bit_length}-bit product has more than #{count} digits" if text.size > count * width(base)

        text.scan(/\d{#{width(base)}}/).reverse.map(&:to_i)
      end

      def carry(coefficients, base, count)
        carry = 0
        digits = Array.new(count) do |k|
          carry, digit = (carry + coefficients.fetch(k, 0).to_i).divmod(base)
          digit
        end
        [digits, carry]
      end

      def mismatches(digits, overflow, expected)
        digits.zip(expected).count { |x, y| x != y } + (overflow.zero? ? 0 : 1)
      end

      def values(digits)
        { 'product_digits' => (digits.rindex(&:nonzero?) || -1) + 1, 'digit_sum' => digits.sum }
      end
    end
  end
end
