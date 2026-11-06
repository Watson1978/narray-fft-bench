# frozen_string_literal: true

require 'bundler/setup'
require 'json'

module FFTBench
  class Error < StandardError; end

  ROOT = File.expand_path('..', __dir__)
  DATA_DIR = ENV.fetch('FFT_BENCH_DATA') { File.join(ROOT, 'data') }
end

require_relative 'fft_bench/numo_fft'
require_relative 'fft_bench/backend'
require_relative 'fft_bench/fixtures'
require_relative 'fft_bench/gates'
require_relative 'fft_bench/timer'
require_relative 'fft_bench/problem'
Dir[File.join(__dir__, 'fft_bench/problems/*.rb')].each { |path| require path }
