#!/usr/bin/env ruby
# frozen_string_literal: true

# Runs the correctness gates of every problem and prints one JSON document.
# Exits 1 if any gate fails. Gates run in fp64 unless DTYPE says otherwise.
#
#   ruby script/gate.rb
#   GPU=1 ruby script/gate.rb --only bignum_mul
#   ruby script/gate.rb --out tmp/gates/numo.json

require 'optparse'

ENV['DTYPE'] ||= 'fp64'
require_relative '../lib/fft_bench'

options = { size: 'small', only: nil, out: nil }
OptionParser.new do |parser|
  parser.banner = 'Usage: ruby script/gate.rb [--size small] [--only a,b] [--out path]'
  parser.on('--size NAME', 'fixture のサイズ (既定 small)') { |v| options[:size] = v }
  parser.on('--only a,b', Array, '回す題材を絞る') { |v| options[:only] = v }
  parser.on('--out PATH', 'JSON を標準出力ではなくファイルに書く') { |v| options[:out] = v }
end.parse!

names = options[:only] || FFTBench::Problem.registry.keys
problems = names.to_h do |name|
  problem = FFTBench::Problem.fetch(name).new
  entry =
    begin
      result = problem.run(problem.fixture(options[:size]))
      gates = problem.gate(result)
      { 'status' => gates.all?(&:pass?) ? 'pass' : 'fail', 'gates' => FFTBench::Gates.report(gates),
        'values' => result.fetch('values', {}) }
    rescue NotImplementedError => e
      { 'status' => 'skipped', 'reason' => "#{e.class}: #{e.message}" }
    end
  [name, entry]
end

report = { 'impl' => 'ruby', 'backend' => FFTBench.backend_name, 'dtype' => FFTBench.dtype_name,
           'size' => options[:size], 'problems' => problems }
json = JSON.pretty_generate(report)
if options[:out]
  File.write(options[:out], "#{json}\n")
else
  puts json
end
exit(problems.values.any? { |entry| entry['status'] == 'fail' } ? 1 : 0)
