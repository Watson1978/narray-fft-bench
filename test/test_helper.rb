# frozen_string_literal: true

ENV['DTYPE'] ||= 'fp64'
$LOAD_PATH.unshift(File.expand_path('../lib', __dir__))

require 'open3'
require 'tmpdir'
require 'test/unit'
require 'fft_bench'

module TestHelper
  ROOT = File.expand_path('..', __dir__)
  PYTHON = ENV.fetch('PYTHON') { File.join(ROOT, 'python/.venv/bin/python') }
  PROBLEMS = %w[whisper_mel taylor_green poisson3d bignum_mul phase_correlation ofdm spectrum].freeze

  def require_fixture(problem, size)
    omit("#{problem}_#{size} not found; run `rake fixtures`") unless FFTBench::Fixtures.exist?(problem, size)
    FFTBench::Fixtures.read(problem, size)
  end

  # GPU stays off for the child: the Python side is checked on numpy here.
  def run_python(*args)
    omit("#{PYTHON} not found; see AGENTS.md (環境)") unless File.executable?(PYTHON)
    out, err, status = Open3.capture3({ 'GPU' => nil }, 'timeout', '300', PYTHON, *args, chdir: ROOT)
    assert(status.success?, "python #{args.join(' ')} failed:\n#{err}")
    out
  end

  def run_ruby(*args)
    out, err, status = Open3.capture3({ 'GPU' => nil, 'DTYPE' => nil }, 'timeout', '300', RbConfig.ruby, *args, chdir: ROOT)
    [out, err, status]
  end

  def sha256(array)
    Digest::SHA256.hexdigest(HM::DFloat.cast(array).to_binary)
  end
end
