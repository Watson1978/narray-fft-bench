# frozen_string_literal: true

require 'numo/narray/alt'
require 'numo/linalg'
HM = Numo

module FFTBench
  # Cumo::FFT first shipped in cumo 0.14.0. It is built into vendor/ for
  # this GPU, apart from the shared gem directory the sibling projects use.
  # GPU=1 loads the build in vendor/ (or wherever CUMO_LIB points).
  DEFAULT_CUMO = 'cumo-0.15.0'

  def self.add_cumo_load_path
    lib = ENV.fetch('CUMO_LIB') { Dir[File.expand_path("../../vendor/#{DEFAULT_CUMO}/gems/cumo-*/lib", __dir__)].first }
    raise Error, 'no cumo build found; see AGENTS.md (cumo の版) or set CUMO_LIB' unless lib && File.directory?(lib)

    $LOAD_PATH.unshift(lib) unless $LOAD_PATH.include?(lib)
    lib
  end

  def self.cumo_so
    $LOADED_FEATURES.grep(%r{/cumo\.so\z}).first
  end
end

if ENV['GPU'].to_s =~ /\A(1|on|true)\z/i
  FFTBench.add_cumo_load_path
  require 'cumo/narray'
  require 'cumo/fft'
  XM = Cumo
  XFT = Cumo::FFT
else
  XM = Numo
  XFT = FFTBench::NumoFFT
end

XF =
  case (dtype = ENV['DTYPE'].to_s.downcase)
  when '', 'fp32', 'sfloat' then XM::SFloat
  when 'fp64', 'dfloat' then XM::DFloat
  else raise FFTBench::Error, "unknown DTYPE #{dtype.inspect}; use fp32 or fp64"
  end
XC = XF == XM::DFloat ? XM::DComplex : XM::SComplex

module FFTBench
  module_function

  def gpu?
    XM.name == 'Cumo'
  end

  def fp64?
    XF == XM::DFloat
  end

  def dtype_name
    fp64? ? 'fp64' : 'fp32'
  end

  def backend_name
    gpu? ? 'cumo' : 'numo'
  end

  def describe
    if gpu?
      cache = Cumo::CUDA::Cufft.plan_cache_info
      "cumo #{Cumo::VERSION} #{cumo_so} / plan cache #{cache[:size]}/#{cache[:capacity]}, #{cache[:misses]} misses"
    else
      "numo-pocketfft #{Numo::Pocketfft::VERSION}"
    end
  end

  def complex?(array)
    [XM::DComplex, XM::SComplex, HM::DComplex, HM::SComplex].any? { |type| array.is_a?(type) }
  end

  # Cumo reductions return a 0-d NArray where Numo returns a Ruby Numeric.
  # Reading one back drains the queue, so never call this inside a hot loop.
  def scalar(value)
    return value if value.is_a?(Numeric)

    value.respond_to?(:extract_cpu) ? value.extract_cpu : value.extract
  end

  def synchronize
    XM::CUDA::Runtime.cudaDeviceSynchronize if gpu?
  end

  # Loads the FFT library and builds a plan for a length no condition uses,
  # so the first timed call pays for its own plans and nothing else.
  def prime
    XFT.fft(XM::DComplex.new(3).seq)
    synchronize
  end

  # Fixtures are always read on the host as Numo::DFloat or Numo::DComplex.
  # This is the one place they cross to the backend and to the working dtype.
  def device(host, dtype = nil)
    dtype ||= complex?(host) ? XC : XF
    array = gpu? ? (complex?(host) ? XM::DComplex : XM::DFloat).from_binary(host.to_binary, host.shape) : host
    array.is_a?(dtype) ? array : dtype.cast(array)
  end

  def host(array)
    return array unless gpu?

    complex = complex?(array)
    bytes = (complex ? XM::DComplex : XM::DFloat).cast(array).to_binary
    (complex ? HM::DComplex : HM::DFloat).from_binary(bytes, array.shape)
  end
end
