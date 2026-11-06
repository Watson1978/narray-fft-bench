#!/usr/bin/env ruby
# frozen_string_literal: true

# Builds data/<problem>_<size>.{bin,json} on the host, in fp64, from a fixed
# seed. This is the only place random numbers are drawn: Numo, Cumo and numpy
# do not share a generator, so every implementation reads these bytes instead.
# Spectral shaping here runs on FFTBench::NumoFFT, on the host.
#
#   ruby script/make_fixtures.rb                     # every problem, small
#   ruby script/make_fixtures.rb --size small bignum_mul
#   ruby script/make_fixtures.rb --force             # rewrite existing files
#
# shim_small holds the inputs and the case list of test/test_numo_fft.rb;
# python/make_shim_refs.py turns it into shim_ref_small with numpy.

require 'optparse'
require_relative '../lib/fft_bench'

module FixtureGenerators
  DFloat = HM::DFloat
  DComplex = HM::DComplex
  NMath = HM::NMath
  FFT = FFTBench::NumoFFT
  MEL_PATH = File.expand_path('whisper_mel_filters.json', __dir__)

  SEEDS = {
    'shim' => 2000, 'whisper_mel' => 2001, 'taylor_green' => 2002, 'poisson3d' => 2003, 'bignum_mul' => 2004,
    'phase_correlation' => 2005, 'ofdm' => 2006, 'spectrum' => 2007
  }.freeze

  SIZES = {
    'shim' => { 'small' => {} },
    'whisper_mel' => {
      'small' => { seconds: 30 },
      'f300' => { seconds: 3 },
      'f30000' => { seconds: 30, repeat: 10 }
    },
    'taylor_green' => {
      'small' => { n: 64, nu: 0.05, dt: 0.01, steps: 100 },
      **[256, 512, 1024, 2048, 4096].to_h { |n| ["n#{n}", { n: n, nu: 0.05, dt: 0.01 }] }
    },
    'poisson3d' => {
      'small' => { n: 64, odd: [31, 32, 33], box: 1.0, sigma: 1.0 / 16 },
      **[128, 256, 512].to_h { |n| ["n#{n}", { n: n, box: 1.0, sigma: 1.0 / 16 }] }
    },
    'bignum_mul' => {
      'small' => { digits: 1000, base: 10_000 },
      **(4..7).to_h { |e| ["n1e#{e}", { digits: 10**e, base: 10_000 }] },
      'n2e7' => { digits: 2 * 10**7, base: 10_000 },
      **(3..6).to_h { |e| ["b10_n1e#{e}", { digits: 10**e, base: 10 }] }
    },
    'phase_correlation' => {
      'small' => { n: 256, odd: 255, shifts: 20, fractional: 5 },
      'n512' => { n: 512, shifts: 4, fractional: 2 },
      'n1024' => { n: 1024, odd: 1023, shifts: 4, fractional: 2 },
      'n2048' => { n: 2048, shifts: 4, fractional: 1 },
      'n4096' => { n: 4096, shifts: 4, fractional: 1 }
    },
    'ofdm' => {
      'small' => { n: 64, symbols: 1000, taps: 5, ebn0_db: [0, 2, 4, 6, 8, 10] },
      **[64, 256, 1024].product([1000, 10_000]).to_h { |n, s| ["n#{n}_s#{s}", { n: n, symbols: s, taps: 5 }] }
    },
    'spectrum' => { 'small' => { short: [256, 420, 257], long: [65_536, 75_600, 65_537], fs: 8192.0,
                                 f0: 8192.0 * 37 / 256, amplitude: 1.0, sigma: 0.5 } }
  }.freeze

  SHIM_INPUTS = {
    'r5' => [5], 'r16' => [16], 'r17' => [17], 'r23' => [2, 3], 'r56' => [5, 6], 'r345' => [3, 4, 5],
    'c5' => [5], 'c9' => [9], 'c16' => [16], 'c17' => [17], 'c56' => [5, 6], 'c345' => [3, 4, 5]
  }.freeze
  NORMS = [nil, 'backward', 'ortho', 'forward'].freeze

  module_function

  def randn(*shape) = DFloat.new(*shape).rand_norm

  # Unit variance: the real and imaginary parts each carry half of it.
  def crandn(*shape)
    DComplex.zeros(*shape).tap do |z|
      z.real = randn(*shape) * Math.sqrt(0.5)
      z.imag = randn(*shape) * Math.sqrt(0.5)
    end
  end

  def unit_rms(x)
    x / Math.sqrt((x * x).mean)
  end

  # |k| on the grid of an n x n rfft2, in integer wavenumbers.
  def wavenumber2(n)
    ky = FFT.fftfreq(n, d: 1.0 / n).reshape(n, 1)
    kx = FFT.rfftfreq(n, d: 1.0 / n).reshape(1, (n / 2) + 1)
    NMath.sqrt((ky * ky) + (kx * kx))
  end

  def shim(_spec)
    arrays = SHIM_INPUTS.to_h { |name, shape| [name, name.start_with?('c') ? crandn(*shape) : randn(*shape)] }
    [arrays, { 'cases' => shim_cases }]
  end

  # Every combination of n / axis / norm on the 1-D functions, of s / axes /
  # norm on the N-D ones, plus the frequencies, the shifts and fftconvolve.
  # s is given with axes only: numpy 2 deprecates s without axes.
  def shim_cases
    cases = []
    add = lambda do |func, *args, **kwargs|
      cases << { 'id' => format('%04d', cases.size), 'func' => func, 'args' => args,
                 'kwargs' => kwargs.compact.transform_keys(&:to_s) }
    end
    { 'fft' => %w[c16 c17 r17 c345 r345], 'ifft' => %w[c16 c17 c345], 'rfft' => %w[r16 r17 r345],
      'irfft' => %w[c9 c16 c345], 'hfft' => %w[c9 c16 c345], 'ihfft' => %w[r16 r17 r345] }.each do |func, inputs|
      inputs.each do |input|
        axes = SHIM_INPUTS[input].size == 1 ? [nil, 0] : [nil, 0, 1]
        axes.product([nil, 1, 7, 8, 32, 33], NORMS).each { |axis, n, norm| add.(func, input, n: n, axis: axis, norm: norm) }
      end
    end
    { 'fftn' => %w[c345 r345], 'ifftn' => %w[c345], 'rfftn' => %w[r345], 'irfftn' => %w[c345],
      'fft2' => %w[c56 c345], 'ifft2' => %w[c56], 'rfft2' => %w[r56 r345], 'irfft2' => %w[c56 c345] }.each do |func, inputs|
      inputs.each do |input|
        ndim = SHIM_INPUTS[input].size
        two = func.end_with?('2')
        axes_list = two ? [nil, ndim == 2 ? [1, 0] : [0, 2]] : [nil, [0, 1], [1, 2], [2, 0], [0, 2, 1], [1]]
        axes_list.each do |axes|
          k = axes ? axes.size : (two ? 2 : ndim)
          [nil, [6, 3, 8].first(k), ([-1] * (k - 1)) + [7]].product(NORMS).each do |s, norm|
            next if s && axes.nil? && !two

            add.(func, input, s: s, axes: axes, norm: norm)
          end
        end
      end
    end
    add.('fftn', 'c345', s: [3, 6], axes: [1, 1])
    add.('ifftn', 'c345', axes: [0, 0], norm: 'ortho')
    add.('rfftn', 'r345', axes: [2, 2])
    add.('rfftn', 'r345', s: [3, 6, 4], axes: [1, 1, 2])
    add.('irfftn', 'c345', s: [3, 6, 8], axes: [1, 1, 2])
    %w[fftfreq rfftfreq].each do |func|
      [1, 2, 7, 8, 400].product([nil, 0.1, 1.0 / 16_000]).each { |n, d| add.(func, n, d: d) }
    end
    %w[fftshift ifftshift].each do |func|
      { 'r16' => [nil, 0], 'r17' => [nil, -1], 'r56' => [nil, 0, 1, [0, 1]],
        'c345' => [nil, 1, [0, 2], [1, 1], [-1, 0]] }.each do |input, axes_list|
        axes_list.each { |axes| add.(func, input, axes: axes) }
      end
    end
    [%w[r17 r5], %w[c17 r5], %w[r5 c5], %w[r56 r23]].each { |a, b| add.('fftconvolve', a, b) }
    cases
  end

  # The waveform narray-llm's Whisper tests run on (python/whisper_dump.py
  # there): a chirp under an envelope plus a steady tone, rounded to fp32.
  def whisper_mel(spec)
    table = JSON.parse(File.read(MEL_PATH))
    rate = table.fetch('sampling_rate')
    seconds = spec[:seconds]
    t = DFloat.new(rate * seconds).seq / rate
    sweep = NMath.sin(2 * Math::PI * (200.0 + (60.0 * t)) * t)
    tone = 0.3 * NMath.sin(2 * Math::PI * 440.0 * t)
    envelope = 0.5 * (1.0 - NMath.cos(2 * Math::PI * t / seconds))
    waveform = DFloat.cast(HM::SFloat.cast((sweep + tone) * envelope))
    repeat = spec.fetch(:repeat, 1)
    waveform = waveform.tile(repeat) if repeat > 1
    attrs = table.slice('sampling_rate', 'n_fft', 'hop_length', 'feature_size')
                 .merge('n_samples' => waveform.size, 'nb_max_frames' => waveform.size / table.fetch('hop_length'))
    [{ 'waveform' => waveform, 'mel_filters' => DFloat.cast(table.fetch('mel_filters')) }, attrs]
  end

  # Taylor-Green 2 sin x sin y, and a random field whose energy spectrum
  # falls as k^-5 (vorticity amplitude as k^-2), band-limited to the 2/3
  # rule so dealiasing leaves it alone.
  def taylor_green(spec)
    n = spec[:n]
    x = DFloat.new(n).seq * (2 * Math::PI / n)
    taylor = 2 * NMath.sin(x.reshape(n, 1)) * NMath.sin(x.reshape(1, n))
    k = wavenumber2(n)
    shape = (k**-2) * (k <= n / 3.0)
    shape[0, 0] = 0
    random = unit_rms(FFT.irfft2(FFT.rfft2(randn(n, n)) * shape, s: [n, n]))
    [spec[:steps] ? { 'omega_taylor' => taylor, 'omega_random' => random } : { 'omega_random' => random }, {}]
  end

  # A unit charge, a Gaussian of width sigma, at the middle of the box.
  def gaussian_charge(shape, box, sigma)
    r2 = shape.each_with_index.map do |n, axis|
      c = (DFloat.new(n).seq * (box / n)) - (box / 2)
      (c * c).reshape(*shape.each_index.map { |i| i == axis ? n : 1 })
    end.reduce(:+)
    NMath.exp(-r2 / (2 * sigma * sigma)) / (((2 * Math::PI) * sigma * sigma)**1.5)
  end

  # Grid offsets from the charge at index n / 2, all under L / 4, along an
  # axis, a face diagonal and a body diagonal.
  POISSON_POINTS = [[2, 0, 0], [4, 0, 0], [8, 0, 0], [12, 0, 0], [15, 0, 0], [2, 2, 0], [5, 5, 0], [10, 10, 0],
                    [2, 2, 2], [4, 4, 4], [8, 8, 8]].freeze

  # The periodic potential of a unit point charge over a neutralizing
  # background (nabla^2 psi = -(delta - 1 / L^3)), by Ewald summation, up to
  # a constant: erfc in real space plus a Gaussian-damped Fourier series.
  def ewald(r, box, alpha: 5.0 / box, images: 2, modes: 10)
    real = 0.0
    (-images..images).to_a.repeated_permutation(3) do |shift|
      d = Math.sqrt(r.zip(shift).sum { |x, s| (x + (s * box))**2 })
      real += Math.erfc(alpha * d) / (4 * Math::PI * d)
    end
    reciprocal = 0.0
    (-modes..modes).to_a.repeated_permutation(3) do |m|
      next if m.all?(&:zero?)

      k = m.map { |i| 2 * Math::PI * i / box }
      k2 = k.sum { |x| x * x }
      reciprocal += Math.exp(-k2 / (4 * alpha * alpha)) / k2 * Math.cos(k.zip(r).sum { |a, b| a * b })
    end
    real + (reciprocal / (box**3))
  end

  # The periodic, zero-mean potential of the Gaussian charge (nabla^2 phi =
  # -rho), up to a constant: the point-charge lattice plus the smearing of
  # the central charge, erf(r / (sqrt 2 sigma)) / (4 pi r) - 1 / (4 pi r).
  # The images sit at least 3 L / 4 away, where the smearing is below 1e-30.
  def periodic_gaussian_potential(r, box, sigma)
    d = Math.sqrt(r.sum { |x| x * x })
    ewald(r, box) + ((Math.erf(d / (Math.sqrt(2) * sigma)) - 1) / (4 * Math::PI * d))
  end

  def poisson3d(spec)
    n = spec[:n]
    arrays = { 'rho' => gaussian_charge([n, n, n], spec[:box], spec[:sigma]) }
    return [arrays, {}] unless spec[:odd]

    h = spec[:box] / n
    reference = POISSON_POINTS.map { |p| periodic_gaussian_potential(p.map { |i| i * h }, spec[:box], spec[:sigma]) }
    arrays.merge!('rho_odd' => gaussian_charge(spec[:odd], spec[:box], spec[:sigma]),
                  'points' => DFloat.cast(POISSON_POINTS), 'reference' => DFloat.cast(reference))
    [arrays, {}]
  end

  # Little-endian digits in base B, the top digit nonzero.
  def digits(count, base)
    d = DFloat.new(count).rand(base).floor
    d[-1] = 1 + DFloat.new(1).rand(base - 1).floor[0]
    d
  end

  # The coefficients of the product, exactly, by Kronecker substitution with
  # Ruby's Integer: digit i goes into bits 64i to 64i + 63.
  def exact_convolution(a, b)
    length = a.size + b.size - 1
    bound = [a.size, b.size].min * a.max * b.max
    raise ArgumentError, "coefficients up to #{bound} do not fit in fp64" if bound >= 2**53

    pack = ->(d) { d.to_a.reverse.map { |x| format('%016x', x.to_i) }.join.to_i(16) }
    hex = (pack.(a) * pack.(b)).to_s(16).rjust(16 * length, '0')
    DFloat.cast(hex.scan(/\h{16}/).reverse.map { |h| h.to_i(16) })
  end

  def bignum_mul(spec)
    a = digits(spec[:digits], spec[:base])
    b = digits(spec[:digits], spec[:base])
    [{ 'a' => a, 'b' => b, 'exact' => exact_convolution(a, b) }, { 'base' => spec[:base] }]
  end

  # White noise low-passed by 1 / (1 + (k / k0)^2) with k0 = n / 16 cycles,
  # unit RMS. A Gaussian would push the top bins under the rounding of the
  # image, and their phases would then be noise.
  def texture(n)
    ky = FFT.fftfreq(n, d: 1.0 / n).reshape(n, 1)
    kx = FFT.fftfreq(n, d: 1.0 / n).reshape(1, n)
    filter = 1 / (1 + (((ky * ky) + (kx * kx)) / ((n / 16.0)**2)))
    unit_rms(FFT.ifft2(FFT.fft2(randn(n, n)) * filter).real)
  end

  # Row i of shifts is [axis 0, axis 1]: J = roll(I, shift). The fractional
  # images are I with the phase of every bin turned by the shift.
  def phase_correlation(spec)
    n = spec[:n]
    image = texture(n)
    odd = texture(spec[:odd]) if spec[:odd]
    span = n / 4
    shifts = DFloat.new(spec[:shifts], 2).rand((2 * span) - 1).floor - (span - 1)
    fractional = (DFloat.new(spec[:fractional], 2).rand * ((2 * span) - 2)) - (span - 1)
    k = FFT.fftfreq(n)
    spectrum = FFT.fft2(image)
    frac_images = DFloat.zeros(spec[:fractional], n, n)
    spec[:fractional].times do |i|
      phase = (k.reshape(n, 1) * fractional[i, 0]) + (k.reshape(1, n) * fractional[i, 1])
      turn = DComplex.zeros(n, n).tap { |z| z.real = NMath.cos(-2 * Math::PI * phase); z.imag = NMath.sin(-2 * Math::PI * phase) }
      frac_images[i, true, true] = FFT.ifft2(spectrum * turn).real
    end
    arrays = { 'image' => image, 'shifts' => shifts, 'frac_shifts' => fractional, 'frac_images' => frac_images }
    [odd ? arrays.merge('image_odd' => odd) : arrays, {}]
  end

  # QPSK bits, a multipath channel shorter than the cyclic prefix with unit
  # energy, and unit-variance complex noise the problem scales per Eb/N0.
  # The bench sizes have no noise.
  def ofdm(spec)
    n = spec[:n]
    cp = n / 8
    bits = DFloat.new(spec[:symbols], 2 * n).rand(2).floor
    taps = crandn(spec[:taps]) * NMath.exp(-DFloat.new(spec[:taps]).seq / 2)
    taps /= Math.sqrt(((taps.real * taps.real) + (taps.imag * taps.imag)).sum)
    arrays = { 'bits' => bits, 'channel' => taps }
    arrays['noise'] = crandn(spec[:symbols], n + cp) if spec[:ebn0_db]
    [arrays, { 'cp' => cp }]
  end

  # r[k] = sum of x[n] x[n + k], summed term by term.
  def direct_autocorrelation(x)
    n = x.size
    DFloat.cast(Array.new(n) { |k| (x[0...(n - k)] * x[k...n]).sum })
  end

  # A sine of amplitude A at f0 plus white noise of standard deviation sigma.
  # f0 sits on a bin of every segment length that divides fs / 32.
  def spectrum(spec)
    signal = lambda do |length|
      t = DFloat.new(length).seq / spec[:fs]
      (spec[:amplitude] * NMath.sin(2 * Math::PI * spec[:f0] * t)) + (spec[:sigma] * randn(length))
    end
    arrays = {}
    %w[pow2 mixed prime].zip(spec[:short], spec[:long]).each do |kind, short, long|
      arrays["short_#{kind}"] = signal.(short)
      arrays["short_#{kind}_direct"] = direct_autocorrelation(arrays["short_#{kind}"])
      arrays[kind] = signal.(long)
    end
    [arrays, {}]
  end

  def build(problem, size)
    spec = SIZES.fetch(problem).fetch(size)
    HM::NArray.srand(SEEDS.fetch(problem))
    arrays, attrs = send(problem, spec)
    attrs = attrs.merge('seed' => SEEDS.fetch(problem), 'spec' => spec.transform_keys(&:to_s))
    FFTBench::Fixtures.write(problem, size, arrays, attrs: attrs)
  end
end

if $PROGRAM_NAME == __FILE__
  options = { size: 'small', force: false }
  OptionParser.new do |parser|
    parser.banner = 'Usage: ruby script/make_fixtures.rb [--size small] [--force] [problem ...]'
    parser.on('--size NAME', 'small (既定)。all で題材のサイズすべて') { |v| options[:size] = v }
    parser.on('--force', '既にあるファイルも作り直す') { options[:force] = true }
  end.parse!
  problems = ARGV.empty? ? FixtureGenerators::SIZES.keys : ARGV
  problems.each do |problem|
    sizes = FixtureGenerators::SIZES.fetch(problem)
    names = options[:size] == 'all' ? sizes.keys : [options[:size]]
    names.each do |size|
      unless sizes.key?(size)
        warn "#{problem}: no #{size} size yet"
        next
      end
      if !options[:force] && FFTBench::Fixtures.exist?(problem, size)
        puts "#{problem}_#{size}: exists"
        next
      end
      manifest = FixtureGenerators.build(problem, size)
      puts "#{problem}_#{size}: #{manifest['bytes']} bytes, #{manifest['sha256'][0, 16]}"
    end
  end
end
