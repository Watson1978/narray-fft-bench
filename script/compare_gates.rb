#!/usr/bin/env ruby
# frozen_string_literal: true

# Lines up the gate JSON of several implementations, problem by problem.
# Gate values are errors and may differ; the "values" (energies, shifts,
# BERs) come from the same fixture and the same algorithm, so they must agree.
# Exits 1 when they do not. A gate one implementation lacks is shown as ×.
#
#   ruby script/gate.rb --out tmp/gates/numo.json
#   python/.venv/bin/python python/gate.py --out tmp/gates/numpy.json
#   ruby script/compare_gates.rb tmp/gates/*.json

require 'json'

VALUE_TOL = { 'fp64' => 1e-9, 'fp32' => 1e-3 }.freeze

docs = ARGV.map { |path| JSON.parse(File.read(path)) }
abort 'usage: compare_gates.rb a.json b.json ...' if docs.size < 2

labels = docs.map { |doc| doc['backend'] }
tol = VALUE_TOL.fetch(docs.map { |doc| doc['dtype'] }.max_by { |dtype| VALUE_TOL[dtype] })
disagree = false

fmt = lambda do |value|
  case value
  when nil then '×'
  when Float then format('%.4g', value)
  else value.to_s
  end
end

docs.flat_map { |doc| doc['problems'].keys }.uniq.each do |problem|
  entries = docs.map { |doc| doc['problems'][problem] || { 'status' => 'missing' } }
  puts "## #{problem}", ''
  puts "| | #{labels.join(' | ')} |", "|---|#{'---|' * labels.size}"
  puts "| status | #{entries.map { |entry| entry['status'] }.join(' | ')} |"
  entries.flat_map { |entry| entry.fetch('gates', {}).keys }.uniq.each do |gate|
    puts "| #{gate} | #{entries.map { |entry| fmt.call(entry.dig('gates', gate, 'value')) }.join(' | ')} |"
  end
  entries.flat_map { |entry| entry.fetch('values', {}).keys }.uniq.each do |name|
    values = entries.map { |entry| entry.dig('values', name) }
    present = values.compact
    spread = present.empty? ? 0.0 : (present.max - present.min).abs / [present.map(&:abs).max, 1e-300].max
    mark = spread <= tol ? '' : " **不一致 #{format('%.2g', spread)}**"
    disagree ||= spread > tol
    puts "| #{name} | #{values.map { |value| fmt.call(value) }.join(' | ')}#{mark} |"
  end
  puts
end
exit(disagree ? 1 : 0)
