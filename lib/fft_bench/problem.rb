# frozen_string_literal: true

module FFTBench
  # One subclass per problem (A-G). script/gate.rb and script/bench.rb call
  # these four methods and nothing else.
  class Problem
    @registry = {}

    class << self
      attr_reader :registry

      def register(name)
        Problem.registry[name.to_s] = self
        @problem_name = name.to_s
      end

      attr_reader :problem_name

      def fetch(name)
        Problem.registry.fetch(name.to_s) do
          raise Error, "unknown problem #{name.inspect}; known: #{Problem.registry.keys.join(', ')}"
        end
      end
    end

    def name
      self.class.problem_name
    end

    def fixture(size)
      Fixtures.read(name, size)
    end

    # Returns a Hash of whatever the gates need, arrays still on the device.
    # "values" holds the physical quantities the four implementations must
    # agree on (script/compare_gates.rb).
    def run(_fixture)
      raise NotImplementedError, "#{self.class}#run"
    end

    # Returns an Array of Gates::Gate.
    def gate(_result)
      raise NotImplementedError, "#{self.class}#gate"
    end

    # Returns a callable; one call is one timed step. Work done before it
    # returns (uploads, a fixed spectrum) is outside both timings, plans
    # included. A problem that times more than one thing names them as
    # variants, the first being the default.
    def bench_step(_fixture, _variant = nil)
      raise NotImplementedError, "#{self.class}#bench_step"
    end
  end
end
