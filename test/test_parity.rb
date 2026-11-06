# frozen_string_literal: true

require_relative 'test_helper'

# FFTBench::NumoFFT against Cumo::FFT on the same fp64 input, for the calls
# the problems make. Omitted when cumo does not load.
class TestParity < Test::Unit::TestCase
  include TestHelper

  TOL = 1.6e-14

  def setup
    FFTBench.add_cumo_load_path
    require 'cumo/narray'
    require 'cumo/fft'
  rescue FFTBench::Error, LoadError => e
    omit("cumo does not load: #{e.message}")
  end

  def test_bignum_mul_calls
    fixture = require_fixture('bignum_mul', 'small')
    a = fixture['a']
    b = fixture['b']
    assert_parity('fftconvolve', a, b) { |fft, x, y| fft.fftconvolve(x, y) }
    assert_parity('fftconvolve unequal', a[0...700], b) { |fft, x, y| fft.fftconvolve(x, y) }
    [2048, 1999, 500].each do |m|
      spectrum = FFTBench::NumoFFT.rfft(a, n: m)
      assert_parity("rfft n: #{m}", a) { |fft, x| fft.rfft(x, n: m) }
      assert_parity("irfft n: #{m}", spectrum) { |fft, x| fft.irfft(x, n: m) }
      assert_parity("fft n: #{m}", a) { |fft, x| fft.fft(x, n: m) }
      assert_parity("ifft n: #{m}", FFTBench::NumoFFT.fft(a, n: m)) { |fft, x| fft.ifft(x, n: m) }
    end
  end

  def test_phase_correlation_calls
    fixture = require_fixture('phase_correlation', 'small')
    [fixture['image'], fixture['image_odd']].each do |image|
      n = image.shape[0]
      spectrum = FFTBench::NumoFFT.fft2(image)
      half = FFTBench::NumoFFT.rfft2(image)
      assert_parity("fft2 #{n}", image) { |fft, x| fft.fft2(x) }
      assert_parity("ifft2 #{n}", spectrum) { |fft, x| fft.ifft2(x) }
      assert_parity("rfft2 #{n}", image) { |fft, x| fft.rfft2(x) }
      assert_parity("irfft2 #{n}", half) { |fft, x| fft.irfft2(x, s: [n, n]) }
      assert_parity("fftshift #{n}", image) { |fft, x| fft.fftshift(x) }
      assert_parity("ifftshift #{n}", image) { |fft, x| fft.ifftshift(x) }
      assert_parity("fftshift complex #{n}", spectrum) { |fft, x| fft.fftshift(x) }
    end
  end

  def test_spectrum_calls
    fixture = require_fixture('spectrum', 'small')
    %w[pow2 mixed prime].each do |kind|
      [fixture["short_#{kind}"], fixture[kind]].each do |x|
        n = x.size
        padded = FFTBench::NumoFFT.rfft(x, n: 2 * n)
        half = FFTBench::NumoFFT.rfft(x)
        real_half = half.real[0..(n / 2)]
        assert_parity("rfft n: 2N #{n}", x) { |fft, y| fft.rfft(y, n: 2 * n) }
        assert_parity("irfft n: 2N #{n}", padded) { |fft, y| fft.irfft(y, n: 2 * n) }
        assert_parity("fft n: 2N #{n}", x) { |fft, y| fft.fft(y, n: 2 * n) }
        assert_parity("ifft real #{n}", padded.real) { |fft, y| fft.ifft(y) }
        assert_parity("hfft #{n}", real_half) { |fft, y| fft.hfft(y, n: n) }
        assert_parity("ihfft #{n}", x) { |fft, y| fft.ihfft(y) }
        assert_parity("fftconvolve reverse #{n}", x) { |fft, y| fft.fftconvolve(y, y.reverse) }
      end
      x = fixture[kind]
      [256, 1024, 4096].each do |length|
        count = x.size / length
        segments = x[0...(count * length)].reshape(count, length)
        assert_parity("rfft batch #{length} #{kind}", segments) { |fft, y| fft.rfft(y, axis: -1) }
      end
    end
    assert_parity('rfftfreq', Numo::DFloat[0]) { |fft, _| fft.rfftfreq(256, d: 1.0 / 8192) }
  end

  def test_taylor_green_calls
    fixture = require_fixture('taylor_green', 'small')
    omega = fixture['omega_random']
    n = omega.shape[0]
    spectrum = FFTBench::NumoFFT.rfft2(omega)
    assert_parity('rfft2', omega) { |fft, x| fft.rfft2(x) }
    assert_parity('irfft2 s:', spectrum) { |fft, x| fft.irfft2(x, s: [n, n]) }
    assert_parity('rfft2 transposed', omega) { |fft, x| fft.rfft2(x.transpose, axes: [-1, -2]) }
    assert_parity('rfft2 product', omega) { |fft, x| fft.rfft2(x * x) }
    assert_parity('fftfreq d: 1/n', Numo::DFloat[0]) { |fft, _| fft.fftfreq(n, d: 1.0 / n) }
    assert_parity('rfftfreq d: 1/n', Numo::DFloat[0]) { |fft, _| fft.rfftfreq(n, d: 1.0 / n) }
    assert_parity('fftshift freq', FFTBench::NumoFFT.fftfreq(n, d: 1.0 / n)) { |fft, k| fft.fftshift(k) }
  end

  def test_poisson3d_calls
    fixture = require_fixture('poisson3d', 'small')
    [fixture['rho'], fixture['rho_odd']].each do |rho|
      shape = rho.shape
      half = FFTBench::NumoFFT.rfftn(rho)
      assert_parity("rfftn #{shape}", rho) { |fft, x| fft.rfftn(x) }
      assert_parity("irfftn s: #{shape}", half) { |fft, x| fft.irfftn(x, s: shape) }
      assert_parity("irfftn no s: #{shape}", half) { |fft, x| fft.irfftn(x) }
      assert_parity("fftn #{shape}", rho) { |fft, x| fft.fftn(x) }
      assert_parity("ifftn #{shape}", FFTBench::NumoFFT.fftn(rho)) { |fft, x| fft.ifftn(x) }
      assert_parity("rfftn axes [0, 1] #{shape}", rho) { |fft, x| fft.rfftn(x, axes: [0, 1]) }
      assert_parity("rfftn axes [2, 0, 1] #{shape}", rho) { |fft, x| fft.rfftn(x, axes: [2, 0, 1]) }
      slab = FFTBench::NumoFFT.rfftn(rho, axes: [0, 1])
      assert_parity("irfftn axes [0, 1] #{shape}", slab) { |fft, x| fft.irfftn(x, s: shape[0, 2], axes: [0, 1]) }
      assert_parity("fft axis 2 #{shape}", slab) { |fft, x| fft.fft(x, axis: 2) }
    end
  end

  def test_whisper_mel_calls
    fixture = require_fixture('whisper_mel', 'small')
    waveform = fixture['waveform']
    frames = waveform[0...(3000 * 160)].reshape(3000, 160)
    assert_parity('rfft frames n: 400', frames) { |fft, x| fft.rfft(x, n: 400, axis: -1) }
    assert_parity('rfft frames n: 400 cut', waveform[0...(1000 * 480)].reshape(1000, 480)) { |fft, x| fft.rfft(x, n: 400, axis: -1) }
    assert_parity('rfftfreq 400', Numo::DFloat[0]) { |fft, _| fft.rfftfreq(400, d: 1.0 / 16_000) }
  end

  def test_ofdm_calls
    fixture = require_fixture('ofdm', 'small')
    channel = fixture['channel']
    bits = fixture['bits']
    count, width = bits.shape
    pairs = bits.reshape(count, width / 2, 2)
    symbols = (Numo::DComplex.cast(1 - (2 * pairs[true, true, 0])) +
               (Numo::DComplex.cast(1 - (2 * pairs[true, true, 1])) * Complex::I)) / Math.sqrt(2)
    n = symbols.shape[1]
    [nil, 'backward', 'ortho', 'forward'].each do |norm|
      assert_parity("ifft batch #{norm}", symbols) { |fft, x| fft.ifft(x, norm: norm) }
      assert_parity("fft batch #{norm}", symbols) { |fft, x| fft.fft(x, axis: -1, norm: norm) }
    end
    assert_parity('fft row', symbols[0, true]) { |fft, x| fft.fft(x, norm: 'ortho') }
    assert_parity('fft h n: N', channel) { |fft, h| fft.fft(h, n: n) }
    assert_parity('fftconvolve batch', symbols, channel.reshape(1, channel.size)) { |fft, x, h| fft.fftconvolve(x, h) }
    assert_parity('fftshift axes', symbols) { |fft, x| fft.fftshift(x, axes: [-1]) }
    assert_parity('fftfreq', Numo::DFloat[0]) { |fft, _| fft.fftfreq(n) }
  end

  private

  def assert_parity(label, *inputs)
    expected = yield(FFTBench::NumoFFT, *inputs)
    actual = yield(Cumo::FFT, *inputs.map { |x| cumo(x) })
    assert_equal(expected.class.name.sub('Numo', 'Cumo'), actual.class.name, label)
    actual = numo(actual)
    assert_equal(expected.shape, actual.shape, label)
    assert_operator((actual - expected).abs.max / expected.abs.max, :<=, TOL, label)
  end

  def cumo(host)
    Cumo.const_get(host.class.name.split('::').last).from_binary(host.to_binary, host.shape)
  end

  def numo(device)
    Numo.const_get(device.class.name.split('::').last).from_binary(device.to_binary, device.shape)
  end
end
