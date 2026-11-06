# frozen_string_literal: true

require_relative 'test_helper'

class TestGates < Test::Unit::TestCase
  include TestHelper

  Gates = FFTBench::Gates

  def test_rel_max_is_relative_to_the_largest_reference_element
    assert_equal(0.0, Gates.rel_max(XF[1, -4, 2], XF[1, -4, 2]))
    assert_in_delta(0.5 / 4, Gates.rel_max(XF[1.5, -4, 2], XF[1, -4, 2]), 1e-15)
    assert_raise(FFTBench::Error) { Gates.rel_max(XF[1, 2], XF[1, 2, 3]) }
  end

  def test_mismatches_counts_differing_elements
    assert_equal(0, Gates.mismatches(XF[1, 2, 3], XF[1, 2, 3]))
    assert_equal(2, Gates.mismatches(XF[1, 2, 3], XF[0, 2, 4]))
  end

  def test_parseval_holds_for_every_norm_and_breaks_with_the_wrong_one
    x = XM::NMath.sin(XF.new(4, 15).seq)
    [nil, 'backward', 'ortho', 'forward'].each do |norm|
      spectrum = XFT.fft(x, norm: norm)
      assert_operator(Gates.parseval(x, spectrum, n: 15, norm: norm), :<, 1e-14, norm.inspect)
    end
    assert_operator(Gates.parseval(x, XFT.fft(x), n: 15, norm: 'ortho'), :>, 1)
  end

  def test_parseval_counts_the_one_sided_bins_twice
    [16, 17].each do |n|
      x = XM::NMath.cos(XF.new(3, n).seq)
      half = XFT.rfft(x, norm: 'ortho')
      assert_operator(Gates.parseval(x, half, n: n, norm: 'ortho', half: n), :<, 1e-14, n)
      assert_operator(Gates.parseval(x, half, n: n, norm: 'ortho'), :>, 0.1, n)
    end
  end

  def test_gate_pass_and_report
    gates = [Gates.le('small', 1e-13, 1e-12), Gates.le('large', 1e-8, 1e-12), Gates.eq('shift', 3, 3),
             Gates.le('edge', 0.5, 0.5), Gates.lt('half', 0.5, 0.5)]
    assert_equal([true, false, true, true, false], gates.map(&:pass?))
    assert_equal({ 'value' => 1e-8, 'tol' => 1e-12, 'op' => 'le', 'pass' => false }, Gates.report(gates)['large'])
  end

  def test_default_tol_follows_the_dtype
    assert_equal('fp64', FFTBench.dtype_name)
    assert_equal(1e-12, Gates.tol)
    assert_equal(1e-5, Gates.tol('fp32'))
  end

  def test_steady_returns_rounds_and_per_step
    calls = 0
    result = FFTBench::Timer.measure(rounds: 3, warmup: 2, inner: 4) { calls += 1 }
    assert_equal('steady', result.mode)
    assert_equal(3, result.times.size)
    assert_equal(20, calls)
    assert_in_delta(result.best / 4, result.per_step, 1e-15)
  end

  def test_first_times_one_call_with_no_warm_up
    calls = 0
    result = FFTBench::Timer.first { calls += 1 }
    assert_equal(['first', 1, 1], [result.mode, result.times.size, calls])
  end
end
