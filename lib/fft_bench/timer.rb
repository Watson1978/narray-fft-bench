# frozen_string_literal: true

module FFTBench
  # Two ways to time a step, never mixed in one table.
  #
  # steady: best-of-N after warm-up, so every plan is already cached. Nothing
  # synchronizes inside the timed region; the device is drained once at its
  # end, so every interval includes all the work it queued.
  #
  # first: the first call of the step in the process, plan creation and
  # workspace allocation included. FFTBench.prime runs before it so the
  # library load and the CUDA context are not in it.
  module Timer
    MODES = %w[steady first].freeze

    Result = Struct.new(:mode, :times, :inner) do
      def best = times.min
      def per_step = best / inner
    end

    module_function

    def measure(rounds: 3, warmup: 1, inner: 1, &step)
      raise Error, 'rounds must be at least 1' if rounds < 1

      warmup.times { inner.times(&step) }
      FFTBench.synchronize
      times = Array.new(rounds) { interval(inner, &step) }
      Result.new('steady', times, inner)
    end

    def first(&step)
      FFTBench.prime
      Result.new('first', [interval(1, &step)], 1)
    end

    def interval(inner, &step)
      started = Process.clock_gettime(Process::CLOCK_MONOTONIC)
      inner.times(&step)
      FFTBench.synchronize
      Process.clock_gettime(Process::CLOCK_MONOTONIC) - started
    end
  end
end
