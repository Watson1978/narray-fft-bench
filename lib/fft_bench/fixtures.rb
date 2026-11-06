# frozen_string_literal: true

require 'digest'

module FFTBench
  # A fixture is two files under data/: <problem>_<size>.bin holds every array
  # as little-endian fp64 in row-major order, back to back, and
  # <problem>_<size>.json says where each one starts and what shape it has.
  # A complex array is stored as two real blocks, its real part and then its
  # imaginary part (entries with "part": "re" / "im"), so neither reader
  # depends on how a complex type lays out its bytes. Ruby and Python both
  # read the .bin and must see the same bytes.
  module Fixtures
    Fixture = Struct.new(:problem, :size, :arrays, :attrs, :manifest) do
      def [](name)
        arrays.fetch(name.to_s) { raise Error, "#{problem}_#{size} has no array #{name}" }
      end

      def attr(name)
        attrs.fetch(name.to_s) { raise Error, "#{problem}_#{size} has no attribute #{name}" }
      end

      def device(name, dtype = nil)
        FFTBench.device(self[name], dtype)
      end
    end

    module_function

    def base_path(problem, size, dir: DATA_DIR)
      File.join(dir, "#{problem}_#{size}")
    end

    def exist?(problem, size, dir: DATA_DIR)
      base = base_path(problem, size, dir: dir)
      File.exist?("#{base}.bin") && File.exist?("#{base}.json")
    end

    def parts(array)
      return [[nil, HM::DFloat.cast(array)]] unless FFTBench.complex?(array)

      complex = HM::DComplex.cast(array)
      [['re', complex.real], ['im', complex.imag]]
    end

    def write(problem, size, arrays, attrs: {}, dir: DATA_DIR)
      raise Error, 'fixtures are written little-endian; this host is big-endian' if [1].pack('S') != [1].pack('S<')

      base = base_path(problem, size, dir: dir)
      offset = 0
      entries = []
      File.open("#{base}.bin", 'wb') do |f|
        arrays.each do |name, array|
          parts(array).each do |part, values|
            bytes = values.to_binary
            f.write(bytes)
            entry = { 'name' => name.to_s, 'shape' => array.shape, 'offset' => offset,
                      'sha256' => Digest::SHA256.hexdigest(bytes) }
            entry['part'] = part if part
            entries << entry
            offset += bytes.bytesize
          end
        end
      end
      manifest = { 'problem' => problem.to_s, 'size' => size.to_s, 'dtype' => '<f8', 'order' => 'C',
                   'bytes' => offset, 'sha256' => Digest::SHA256.file("#{base}.bin").hexdigest,
                   'arrays' => entries, 'attrs' => attrs.transform_keys(&:to_s) }
      File.write("#{base}.json", "#{JSON.pretty_generate(manifest)}\n")
      manifest
    end

    def read(problem, size, dir: DATA_DIR)
      base = base_path(problem, size, dir: dir)
      raise Error, "#{base}.bin not found; run `rake fixtures`" unless exist?(problem, size, dir: dir)

      manifest = JSON.parse(File.read("#{base}.json"))
      blocks = manifest['arrays'].group_by { |entry| entry['name'] }
      arrays = blocks.to_h do |name, entries|
        values = entries.to_h { |entry| [entry['part'], read_block(base, entry)] }
        next [name, values.fetch(nil)] if values.keys == [nil]
        raise Error, "#{base}.json: #{name} has parts #{values.keys.inspect}" unless values.keys.sort == %w[im re]

        [name, HM::DComplex.zeros(*entries.first['shape']).tap { |c| c.real = values['re']; c.imag = values['im'] }]
      end
      Fixture.new(problem.to_s, size.to_s, arrays, manifest['attrs'], manifest)
    end

    def read_block(base, entry)
      count = entry['shape'].inject(1, :*)
      bytes = File.binread("#{base}.bin", count * 8, entry['offset'])
      raise Error, "#{base}.bin is truncated at #{entry['name']}" if count.positive? && bytes.to_s.bytesize != count * 8

      HM::DFloat.from_binary(bytes.to_s, entry['shape'])
    end
  end
end
