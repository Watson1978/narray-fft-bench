# frozen_string_literal: true

module FFTBench
  # The shared gate measures. Relative errors are max-norm:
  # ‖x − x_ref‖_max / ‖x_ref‖_max.
  module Gates
    DEFAULT_TOL = { 'fp64' => 1e-12, 'fp32' => 1e-5 }.freeze

    Gate = Struct.new(:name, :value, :tol, :op) do
      def pass?
        case op
        when :le then value <= tol
        when :lt then value < tol
        when :ge then value >= tol
        when :eq then value == tol
        else raise Error, "unknown gate op #{op.inspect}"
        end
      end

      def to_h
        { 'value' => value, 'tol' => tol, 'op' => op.to_s, 'pass' => pass? }
      end
    end

    module_function

    def tol(dtype = FFTBench.dtype_name)
      DEFAULT_TOL.fetch(dtype)
    end

    def le(name, value, tol) = Gate.new(name.to_s, Float(value), Float(tol), :le)
    def lt(name, value, tol) = Gate.new(name.to_s, Float(value), Float(tol), :lt)
    def ge(name, value, tol) = Gate.new(name.to_s, Float(value), Float(tol), :ge)
    def eq(name, value, expected) = Gate.new(name.to_s, value, expected, :eq)

    def max_abs(a)
      FFTBench.scalar(a.abs.max)
    end

    def rel_max(actual, expected)
      raise Error, "shape differs: #{actual.shape} vs #{expected.shape}" if actual.shape != expected.shape

      max_abs(actual - expected) / max_abs(expected)
    end

    # The number of elements that differ, compared on the host. For results
    # that have to come out exactly: digits, bits, shifts, tokens.
    def mismatches(actual, expected)
      a = FFTBench.host(actual)
      e = FFTBench.host(expected)
      raise Error, "shape differs: #{a.shape} vs #{e.shape}" if a.shape != e.shape

      a.ne(e).count_true
    end

    def sum_abs2(x)
      squares = FFTBench.complex?(x) ? (x.real * x.real) + (x.imag * x.imag) : x * x
      FFTBench.scalar(squares.sum)
    end

    # Parseval: Σ|x|² = c Σ|X|² with c = 1/n, 1 and n for the backward, ortho
    # and forward norms, n being the product of the transformed lengths.
    # half: the real length of the last transformed axis when the spectrum is
    # one-sided along it (the last axis of the array), whose bins other than
    # DC and Nyquist then count twice. Answers the relative discrepancy.
    def parseval(x, spectrum, n:, norm: nil, half: nil)
      energy = sum_abs2(x)
      c = { nil => 1.0 / n, 'backward' => 1.0 / n, 'ortho' => 1.0, 'forward' => Float(n) }.fetch(norm&.to_s)
      (energy - (c * spectral_energy(spectrum, half))).abs / energy
    end

    def spectral_energy(spectrum, half)
      total = sum_abs2(spectrum)
      return total if half.nil?

      edges = [0]
      edges << (half / 2) if half.even? && half > 1
      (2 * total) - edges.sum { |k| sum_abs2(spectrum[false, k]) }
    end

    def report(gates)
      gates.to_h { |gate| [gate.name, gate.to_h] }
    end
  end
end
