# frozen_string_literal: true

require_relative 'test_helper'
require 'digest'

class TestFixtures < Test::Unit::TestCase
  include TestHelper

  def test_round_trip_keeps_bytes_shapes_and_attrs
    Dir.mktmpdir do |dir|
      a = HM::DFloat.new(3, 4).seq / 7
      z = HM::DComplex.new(2, 3).seq.tap { |c| c.imag = -HM::DFloat.new(2, 3).seq / 3 }
      FFTBench::Fixtures.write('probe', 'small', { 'a' => a, 'z' => z }, attrs: { 'x' => 0.1, 'n' => 3 }, dir: dir)
      fixture = FFTBench::Fixtures.read('probe', 'small', dir: dir)
      assert_equal(a.to_binary, fixture['a'].to_binary)
      assert_kind_of(HM::DComplex, fixture['z'])
      assert_equal([2, 3], fixture['z'].shape)
      assert_equal(z.to_binary, fixture['z'].to_binary)
      assert_equal([nil, 're', 'im'], fixture.manifest['arrays'].map { |entry| entry['part'] })
      assert_equal(0.1, fixture.attr('x'))
      assert_equal(3, fixture.attr('n'))
    end
  end

  def test_read_rejects_a_truncated_file
    Dir.mktmpdir do |dir|
      FFTBench::Fixtures.write('probe', 'small', { 'a' => HM::DFloat.new(8).seq }, dir: dir)
      path = File.join(dir, 'probe_small.bin')
      File.binwrite(path, File.binread(path, 40))
      assert_raise(FFTBench::Error) { FFTBench::Fixtures.read('probe', 'small', dir: dir) }
    end
  end

  def test_device_casts_to_the_working_dtype
    fixture = require_fixture('ofdm', 'small')
    noise = fixture.device('noise')
    assert_kind_of(XC, noise)
    assert_kind_of(XF, fixture.device('bits'))
    assert_equal(fixture['noise'].shape, noise.shape)
    assert_equal(fixture['noise'].to_binary, FFTBench.host(fixture.device('noise', XM::DComplex)).to_binary)
  end

  def test_bignum_mul_exact_is_the_direct_convolution
    fixture = require_fixture('bignum_mul', 'small')
    a = fixture['a']
    b = fixture['b']
    direct = HM::DFloat.zeros(a.size + b.size - 1)
    a.size.times { |i| direct[i...(i + b.size)] += a[i] * b }
    assert_equal(direct.to_a, fixture['exact'].to_a)
  end

  ['shim', *PROBLEMS].each do |problem|
    define_method("test_#{problem}_small_matches_its_manifest") do
      fixture = require_fixture(problem, 'small')
      fixture.manifest['arrays'].each do |entry|
        array = fixture[entry['name']]
        part = { 're' => :real, 'im' => :imag }[entry['part']]
        assert_equal(entry['shape'], array.shape, entry['name'])
        assert_equal(entry['sha256'], sha256(part ? array.send(part) : array), "#{entry['name']} #{entry['part']}")
      end
    end

    define_method("test_#{problem}_small_reads_the_same_in_python") do
      fixture = require_fixture(problem, 'small')
      python = JSON.parse(run_python('python/fixtures.py', 'digest', problem, 'small'))
      ruby = fixture.arrays.transform_values do |array|
        if FFTBench.complex?(array)
          { 'shape' => array.shape, 're' => sha256(array.real), 'im' => sha256(array.imag) }
        else
          { 'shape' => array.shape, 'sha256' => sha256(array) }
        end
      end
      assert_equal(ruby, python)
    end
  end
end
