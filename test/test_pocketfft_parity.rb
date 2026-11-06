# frozen_string_literal: true

require_relative 'test_helper'

# Cumo::Pocketfft against Numo::Pocketfft: the returned class and shape, the
# raised exception, and the values. Omitted when cumo does not load.
class TestPocketfftParity < Test::Unit::TestCase
  include TestHelper

  TOL = { double: 5e-15, single: 7.5e-7 }.freeze
  TYPES = %w[SFloat DFloat SComplex DComplex Int32].freeze
  FUNCTIONS = %i[fft ifft fft2 ifft2 fftn ifftn rfft irfft rfft2 irfft2 rfftn irfftn].freeze
  SHAPES = [[7], [8], [1], [2], [5, 6], [1, 1], [4, 1], [3, 4, 5], [0], [0, 3]].freeze
  PAIRS = [[[7], [8]], [[1], [1]], [[1], [5]], [[1, 1], [1, 1]], [[3, 4], [2, 5]], [[2, 3, 4], [3, 2, 2]],
           [[3], [3, 1]], [[0], [3]], [[], []]].freeze

  def setup
    FFTBench.add_cumo_load_path
    require 'cumo/narray'
    require 'cumo/pocketfft'
  rescue FFTBench::Error, LoadError => e
    omit("cumo does not load: #{e.message}")
  end

  def test_transforms
    FUNCTIONS.product(TYPES, SHAPES).each do |function, type, shape|
      x = input(type, shape)
      assert_parity("#{function} #{type} #{shape}") { |mod, convert| mod.public_send(function, convert.call(x)) }
    end
  end

  def test_fftconvolve
    TYPES.product(TYPES, PAIRS).each do |ta, tb, (sa, sb)|
      a = input(ta, sa)
      b = input(tb, sb, 0.5)
      assert_parity("fftconvolve #{ta} #{sa} #{tb} #{sb}") { |mod, convert| mod.fftconvolve(convert.call(a), convert.call(b)) }
    end
  end

  def test_non_narray_input
    assert_parity('fft Array') { |mod, _| mod.fft([1.0, 2.0]) }
    assert_parity('fftconvolve Array') { |mod, _| mod.fftconvolve([1.0, 2.0], [1.0]) }
  end

  def test_zero_dimensional_input_raises_dimension_error
    %i[fftn ifftn rfftn irfftn].each do |function|
      assert_raise(Cumo::NArray::DimensionError, function.to_s) { Cumo::Pocketfft.public_send(function, Cumo::DFloat.cast(2.5)) }
    end
  end

  private

  def input(type, shape, phase = 0.0)
    size = shape.reduce(1, :*)
    return Numo.const_get(type).zeros(*shape) if size.zero?

    x = Numo::DFloat.new(size).seq
    real = Numo::NMath.sin((x * 1.3) + 0.2 + phase) * 2
    host =
      case type
      when 'Int32' then Numo::Int32.cast((x % 7) - 3)
      when /Complex/ then Numo.const_get(type).cast(real + (Numo::NMath.cos((x * 0.7) + phase) * Complex::I))
      else Numo.const_get(type).cast(real)
      end
    shape.empty? ? Numo.const_get(type).cast(host[0]) : host.reshape(*shape)
  end

  def assert_parity(label)
    expected = outcome { yield(Numo::Pocketfft, ->(x) { x }) }
    actual = outcome { yield(Cumo::Pocketfft, ->(x) { x.is_a?(Numo::NArray) ? cumo(x) : x }) }
    if expected.is_a?(Exception) || actual.is_a?(Exception)
      assert_equal(describe(expected), describe(actual).gsub('Cumo', 'Numo'), label)
      return
    end
    assert_equal(expected.class.name.sub('Numo', 'Cumo'), actual.class.name, label)
    actual = numo(actual)
    assert_equal(expected.shape, actual.shape, label)
    return if expected.empty?

    tol = TOL[expected.is_a?(Numo::SFloat) || expected.is_a?(Numo::SComplex) ? :single : :double]
    assert_operator(Float((actual - expected).abs.max) / Float(expected.abs.max), :<=, tol, label)
  end

  def outcome
    yield
  rescue StandardError => e
    e
  end

  def describe(result)
    result.is_a?(Exception) ? "#{result.class}: #{result.message}" : "returned #{result.class}"
  end

  def cumo(host)
    Cumo.const_get(host.class.name.split('::').last).from_binary(host.to_binary, host.shape)
  end

  def numo(device)
    Numo.const_get(device.class.name.split('::').last).from_binary(device.to_binary, device.shape)
  end
end
