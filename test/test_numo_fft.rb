# frozen_string_literal: true

require_relative 'test_helper'

# FFTBench::NumoFFT against numpy and scipy, case by case. The inputs and the
# case list are data/shim_small (script/make_fixtures.rb), the references
# data/shim_ref_small (python/make_shim_refs.py).
class TestNumoFFT < Test::Unit::TestCase
  include TestHelper

  # The worst seen is 7.5e-16 (ifft2); this is ten times that, rounded up.
  # The frequencies and the shifts have to match bit for bit.
  TOL = 1e-14
  EXACT = %w[fftfreq rfftfreq fftshift ifftshift].freeze

  def self.cases
    @cases ||= FFTBench::Fixtures.exist?('shim', 'small') ? FFTBench::Fixtures.read('shim', 'small').attr('cases') : []
  end

  def shim
    @shim ||= require_fixture('shim', 'small')
  end

  def refs
    omit('shim_ref_small not found; run `rake fixtures` with the venv') unless FFTBench::Fixtures.exist?('shim_ref', 'small')
    @refs ||= FFTBench::Fixtures.read('shim_ref', 'small')
  end

  def call(entry)
    args = entry['args'].map { |arg| arg.is_a?(String) ? shim[arg] : arg }
    FFTBench::NumoFFT.public_send(entry['func'], *args, **entry['kwargs'].transform_keys(&:to_sym))
  end

  def test_the_references_come_from_these_inputs
    assert_equal(shim.manifest['sha256'], refs.attr('shim_sha256'))
  end

  cases.group_by { |entry| entry['func'] }.each do |func, entries|
    define_method("test_#{func}_matches_numpy") do
      failures = entries.filter_map do |entry|
        got = call(entry)
        want = refs[entry['id']]
        label = "#{entry['id']} #{entry['args'].inspect} #{entry['kwargs']}"
        next "#{label}: #{got.class} #{got.shape} vs #{want.class} #{want.shape}" if got.class != want.class || got.shape != want.shape

        if EXACT.include?(func)
          "#{label}: not bit for bit" unless got.to_binary == want.to_binary
        else
          error = FFTBench::Gates.rel_max(got, want)
          "#{label}: #{error}" unless error <= TOL
        end
      end
      assert_empty(failures, "#{failures.size} of #{entries.size} cases")
    end
  end

  def test_the_output_is_double_precision_whatever_the_input
    x = HM::SFloat.new(8).seq
    assert_kind_of(HM::DComplex, FFTBench::NumoFFT.rfft(x))
    assert_kind_of(HM::DFloat, FFTBench::NumoFFT.irfft(FFTBench::NumoFFT.rfft(x)))
    assert_kind_of(HM::DComplex, FFTBench::NumoFFT.fft(HM::SComplex.new(8).seq))
  end

  def test_bad_arguments_raise_like_cumo
    x = HM::DFloat.new(8).seq
    assert_raise(ArgumentError) { FFTBench::NumoFFT.fft(x, norm: 'both') }
    assert_raise(ArgumentError) { FFTBench::NumoFFT.fft(x, n: 0) }
    assert_raise(ArgumentError) { FFTBench::NumoFFT.irfft(HM::DComplex.new(1).seq) }
    assert_raise(TypeError) { FFTBench::NumoFFT.rfft(HM::DComplex.new(8).seq) }
  end
end
