require 'rake/testtask'

ROOT = __dir__
DATA_DIR = ENV.fetch('FFT_BENCH_DATA') { File.join(ROOT, 'data') }
PYTHON = ENV.fetch('PYTHON') { File.join(ROOT, 'python/.venv/bin/python') }
PROBLEMS = %w[whisper_mel taylor_green poisson3d bignum_mul phase_correlation ofdm spectrum].freeze

Rake::TestTask.new(:test) do |t|
  t.libs << 'lib' << 'test'
  t.test_files = FileList['test/test_*.rb']
  t.warning = false
end

['shim', *PROBLEMS].each do |name|
  file(File.join(DATA_DIR, "#{name}_small.bin")) { ruby 'script/make_fixtures.rb', '--size', 'small', name }
end

shim_ref = File.join(DATA_DIR, 'shim_ref_small.bin')
file(shim_ref => File.join(DATA_DIR, 'shim_small.bin')) do
  if File.executable?(PYTHON)
    sh PYTHON, 'python/make_shim_refs.py'
  else
    warn "#{PYTHON} not found; the NumoFFT tests will be omitted (see AGENTS.md 環境)"
  end
end

desc 'Build the small fixtures and the numpy references of the Numo shim'
task fixtures: [*['shim', *PROBLEMS].map { |name| File.join(DATA_DIR, "#{name}_small.bin") }, shim_ref]

task test: :fixtures

desc 'Run every gate on the current backend (GPU=1 for Cumo)'
task(gate: :fixtures) { ruby 'script/gate.rb' }

desc 'Build every fixture size, including the large ones the benches read'
task('fixtures:all') { ruby 'script/make_fixtures.rb', '--size', 'all' }

desc 'Run bench/conditions.tsv (GPU: Cumo / CuPy) into tmp/bench'
task bench: 'fixtures:all' do
  sh 'bench/run.sh', 'bench/conditions.tsv', 'tmp/bench'
  sh PYTHON, 'bench/aggregate.py', 'bench/conditions.tsv', 'tmp/bench'
end

desc 'Run bench/conditions_cpu.tsv (CPU: Numo / numpy) into tmp/bench_cpu'
task 'bench:cpu' => 'fixtures:all' do
  sh 'bench/run.sh', 'bench/conditions_cpu.tsv', 'tmp/bench_cpu'
  sh PYTHON, 'bench/aggregate.py', 'bench/conditions_cpu.tsv', 'tmp/bench_cpu', '--labels', 'Numo,numpy'
end

task default: :test
