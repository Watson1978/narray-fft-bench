# frozen_string_literal: true

require_relative 'test_helper'

class TestScripts < Test::Unit::TestCase
  include TestHelper

  def test_every_problem_passes_its_gates_on_numo_and_numpy_with_the_same_values
    Dir.mktmpdir do |dir|
      ruby_json = File.join(dir, 'numo.json')
      python_json = File.join(dir, 'numpy.json')
      _, err, status = run_ruby('script/gate.rb', '--out', ruby_json)
      assert(status.success?, err)
      run_python('python/gate.py', '--out', python_json)
      ruby = JSON.parse(File.read(ruby_json))
      python = JSON.parse(File.read(python_json))
      assert_equal(%w[ruby numo fp64 small], ruby.values_at('impl', 'backend', 'dtype', 'size'))
      assert_equal(%w[python numpy fp64 small], python.values_at('impl', 'backend', 'dtype', 'size'))
      assert_equal(FFTBench::Problem.registry.keys.sort, ruby['problems'].keys.sort)
      assert_equal(ruby['problems'].keys.sort, python['problems'].keys.sort)
      [ruby, python].each do |report|
        report['problems'].each do |name, entry|
          assert_equal('pass', entry['status'], "#{report['backend']} #{name}: #{entry}")
        end
      end
      out, err, status = run_ruby('script/compare_gates.rb', ruby_json, python_json)
      assert(status.success?, "#{out}#{err}")
    end
  end

  def test_aggregate_puts_steady_and_first_in_separate_tables
    omit("#{PYTHON} not found") unless File.executable?(PYTHON)
    Dir.mktmpdir do |dir|
      File.write(File.join(dir, 'conditions.tsv'), "# header\nwarm\tsteady\truby x\tpython y\ncold\tfirst\truby x\tpython y\n")
      rows = (0..3).flat_map do |round|
        [[round, 'ruby', 10.0], [round, 'python', 20.0 + round], [round, 'control', 10.0 + ((-1)**round)]]
      end
      %w[warm cold].each { |name| File.write(File.join(dir, "#{name}.tsv"), rows.map { |r| "#{r.join("\t")}\n" }.join) }
      out, err, status = Open3.capture3('timeout', '60', PYTHON, 'bench/aggregate.py',
                                        File.join(dir, 'conditions.tsv'), dir, chdir: ROOT)
      assert(status.success?, err)
      lines = out.lines.map(&:chomp).reject(&:empty?)
      assert_equal(['| 定常 |', '| warm |', '| 初回 |', '| cold |'], lines.values_at(0, 2, 3, 5).map { |line| line[/\A\| \S+ \|/] })
      assert_equal('| cold | 10.000 | 22.000 | 2.200 3/3 p=0.25 | 0.900 1/3 p=1 またぐ | 0.0% | 9.5% 要確認 |', lines.last)
    end
  end

  def test_bench_prints_the_mode_it_ran
    omit('bench.rb needs a problem; the first one lands in the D stage') if FFTBench::Problem.registry.empty?

    name = FFTBench::Problem.registry.keys.first
    %w[steady first].each do |mode|
      out, err, status = run_ruby('script/bench.rb', '--problem', name, '--mode', mode)
      assert(status.success?, err)
      columns = out.chomp.split("\t")
      assert_equal([9, mode], [columns.size, columns.last])
    end
  end

  def test_run_sh_stops_on_a_bad_mode
    Dir.mktmpdir do |dir|
      File.write(File.join(dir, 'conditions.tsv'), "probe\tsometimes\ttrue\ttrue\n")
      out, _, status = Open3.capture3('timeout', '60', 'bench/run.sh', File.join(dir, 'conditions.tsv'), File.join(dir, 'out'),
                                      chdir: ROOT)
      refute(status.success?)
      assert_match(/the mode column is 'sometimes'/, out)
    end
  end
end
