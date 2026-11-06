# frozen_string_literal: true

require 'numo/pocketfft'

module FFTBench
  # The functions and arguments of Cumo::FFT (numpy.fft) on numo-pocketfft.
  # Its public functions take whole 1-D or 2-D arrays only, so this calls the
  # private ext_* functions, which transform the last axis of an array of any
  # shape in one call (AGENTS.md, Numo シム). The N-D functions apply one 1-D
  # transform per axis in numpy's order. Results are DComplex or DFloat.
  module NumoFFT
    POCKETFFT = Numo::Pocketfft
    NORMS = %w[backward ortho forward].freeze

    missing = %i[ext_cfft ext_icfft ext_rfft ext_irfft].reject { |name| POCKETFFT.respond_to?(name, true) }
    raise LoadError, "numo-pocketfft #{Numo::Pocketfft::VERSION} lacks #{missing.join(', ')}" unless missing.empty?

    module_function

    def fft(a, n: nil, axis: -1, norm: nil) = c2c(a, n, axis, norm, false)
    def ifft(a, n: nil, axis: -1, norm: nil) = c2c(a, n, axis, norm, true)
    def rfft(a, n: nil, axis: -1, norm: nil) = r2c(a, n, axis, norm)
    def irfft(a, n: nil, axis: -1, norm: nil) = c2r(a, n, axis, norm)
    def hfft(a, n: nil, axis: -1, norm: nil) = c2r(Numo::DComplex.cast(a).conj, n, axis, swap_norm(norm))
    def ihfft(a, n: nil, axis: -1, norm: nil) = r2c(a, n, axis, swap_norm(norm)).conj

    def fft2(a, s: nil, axes: [-2, -1], norm: nil) = c2cn(a, s, axes, norm, false)
    def ifft2(a, s: nil, axes: [-2, -1], norm: nil) = c2cn(a, s, axes, norm, true)
    def fftn(a, s: nil, axes: nil, norm: nil) = c2cn(a, s, axes, norm, false)
    def ifftn(a, s: nil, axes: nil, norm: nil) = c2cn(a, s, axes, norm, true)
    def rfft2(a, s: nil, axes: [-2, -1], norm: nil) = rfftn(a, s: s, axes: axes, norm: norm)
    def irfft2(a, s: nil, axes: [-2, -1], norm: nil) = irfftn(a, s: s, axes: axes, norm: norm)

    def rfftn(a, s: nil, axes: nil, norm: nil)
      *rest, (axis, n) = cook(a, s, axes, false)
      rest.reverse.reduce(r2c(a, n, axis, norm)) { |x, (ax, m)| c2c(x, m, ax, norm, false) }
    end

    def irfftn(a, s: nil, axes: nil, norm: nil)
      *rest, (axis, n) = cook(a, s, axes, true)
      c2r(rest.reduce(a) { |x, (ax, m)| c2c(x, m, ax, norm, true) }, n, axis, norm)
    end

    def fftfreq(n, d: 1.0)
      k = Numo::DFloat.new(n).seq
      half = ((n - 1) / 2) + 1
      k[half..] -= n if n > half
      k * (1.0 / (n * Float(d)))
    end

    def rfftfreq(n, d: 1.0)
      Numo::DFloat.new((n / 2) + 1).seq * (1.0 / (n * Float(d)))
    end

    def fftshift(x, axes: nil) = roll_half(x, axes, 1)
    def ifftshift(x, axes: nil) = roll_half(x, axes, -1)
    def fftconvolve(a, b) = POCKETFFT.fftconvolve(a, b)

    # numpy's _cook_nd_args: [[axis, length], ...] in the order of axes.
    def cook(a, s, axes, inverse_real)
      axes ||= s.nil? ? (0...a.ndim).to_a : (-s.size...0).to_a
      lengths = s.nil? ? axes.map { |ax| a.shape[ax] } : s.dup
      raise ArgumentError, 'Shape and axes have different lengths.' unless lengths.size == axes.size

      lengths[-1] = 2 * (a.shape[axes[-1]] - 1) if inverse_real && s.nil? && !axes.empty?
      axes.zip(lengths).map { |ax, n| [ax, n == -1 ? a.shape[ax] : n] }
    end

    # numpy's _raw_fftnd: the last axis first.
    def c2cn(a, s, axes, norm, inverse)
      cook(a, s, axes, false).reverse.reduce(a) { |x, (axis, n)| c2c(x, n, axis, norm, inverse) }
    end

    def c2c(a, n, axis, norm, inverse)
      transform(Numo::DComplex.cast(a), n, axis, norm, inverse, inverse ? :ext_icfft : :ext_cfft)
    end

    def r2c(a, n, axis, norm)
      raise TypeError, "expected a real array, got a #{a.class}" if a.is_a?(Numo::DComplex) || a.is_a?(Numo::SComplex)

      transform(Numo::DFloat.cast(a), n, axis, norm, false, :ext_rfft)
    end

    def transform(a, n, axis, norm, inverse, name)
      norm = check_norm(norm)
      along(a, axis) do |x|
        x = fit(x, n || x.shape[-1])
        scale(POCKETFFT.send(name, x), x.shape[-1], norm, inverse)
      end
    end

    # ext_irfft reads the n / 2 + 1 non-negative frequencies from the front of
    # an n-point complex row and answers n real points, whatever n's parity.
    def c2r(a, n, axis, norm)
      norm = check_norm(norm)
      along(Numo::DComplex.cast(a), axis) do |x|
        n ||= 2 * (x.shape[-1] - 1)
        z = fit(x[false, 0...[x.shape[-1], (n / 2) + 1].min], n)
        scale(POCKETFFT.send(:ext_irfft, z), n, norm, true)
      end
    end

    def along(x, axis)
      axis % x.ndim == x.ndim - 1 ? yield(x) : yield(x.swapaxes(axis, -1)).swapaxes(axis, -1).dup
    end

    def fit(x, n)
      raise ArgumentError, "Invalid number of FFT data points (#{n}) specified." unless n.is_a?(Integer) && n >= 1

      m = x.shape[-1]
      return n == m ? x : x[false, 0...n] if n <= m

      x.class.zeros(*x.shape[0...-1], n).tap { |z| z[false, 0...m] = x }
    end

    # ext_cfft and ext_rfft do not scale; ext_icfft and ext_irfft divide by n.
    def scale(out, n, norm, inverse)
      return out if norm == 'backward'

      factor = norm == 'ortho' ? Math.sqrt(n) : n
      out * (inverse ? factor : 1.0 / factor)
    end

    def check_norm(norm)
      name = norm.nil? ? 'backward' : norm.to_s
      NORMS.include?(name) ? name : raise(ArgumentError, "Invalid norm value #{norm.inspect}")
    end

    def swap_norm(norm)
      { 'backward' => 'forward', 'forward' => 'backward', 'ortho' => 'ortho' }.fetch(check_norm(norm))
    end

    def roll_half(x, axes, sign)
      shifts = Hash.new(0)
      Array(axes || (0...x.ndim).to_a).each { |ax| shifts[ax % x.ndim] += sign * (x.shape[ax] / 2) }
      out = shifts.reduce(x) do |acc, (ax, shift)|
        n = acc.shape[ax]
        k = shift % n
        next acc if k.zero?

        index = ->(range) { Array.new(acc.ndim, true).tap { |i| i[ax] = range } }
        acc[*index.((n - k)...n)].concatenate(acc[*index.(0...(n - k))], axis: ax)
      end
      out.equal?(x) ? x.dup : out
    end

    private_class_method :cook, :c2cn, :c2c, :r2c, :transform, :c2r, :along, :fit, :scale, :check_norm, :swap_norm,
                         :roll_half
  end
end
