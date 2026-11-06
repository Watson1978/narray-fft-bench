# frozen_string_literal: true

module FFTBench
  module Problems
    # B. 2D incompressible Navier-Stokes in vorticity form on [0, 2 pi)^2,
    # pseudo-spectral with rfft2 / irfft2. The nonlinear term is the
    # divergence of u omega, dealiased by the 2/3 rule, and time goes by RK4
    # with an integrating factor for the viscosity. Taylor-Green has no
    # nonlinear term and must decay as exp(-2 nu t); the random field at
    # nu = 0 must keep its energy and enstrophy.
    class TaylorGreen < Problem
      register :taylor_green

      TAYLOR_TOL = { 'fp64' => 1e-8, 'fp32' => 5.2e-5 }.freeze
      CONSERVED_TOL = { 'fp64' => 1e-6, 'fp32' => 3.3e-6 }.freeze
      TOL = { 'fp64' => 4.9e-14, 'fp32' => 6e-6 }.freeze

      def run(fixture)
        spec = fixture.attr(:spec)
        omega = fixture.device(:omega_random)
        n = omega.shape[0]
        grid = grid(n, spec['nu'], spec['dt'])
        taylor = fixture.device(:omega_taylor)
        w = XFT.rfft2(taylor)
        spec['steps'].times { w = step(w, grid) }
        exact = taylor * Math.exp(-2 * spec['nu'] * spec['dt'] * spec['steps'])
        taylor = { error: Gates.rel_max(XFT.irfft2(w, s: [n, n]), exact), peak: FFTBench.scalar(XFT.irfft2(w, s: [n, n]).max) }
        random = evolve(omega, grid(n, 0.0, spec['dt']), spec['steps'])
        { 'taylor' => taylor, 'random' => random, 'checks' => checks(omega, n), 'values' => values(taylor, random) }
      end

      def gate(result)
        dtype = FFTBench.dtype_name
        tol = TOL.fetch(dtype)
        r = result['random']
        gates = [Gates.le('taylor.error', result['taylor'][:error], TAYLOR_TOL.fetch(dtype)),
                 Gates.le('random.energy_drift', drift(r[:energy]), CONSERVED_TOL.fetch(dtype)),
                 Gates.le('random.enstrophy_drift', drift(r[:enstrophy]), CONSERVED_TOL.fetch(dtype)),
                 Gates.le('random.parseval', r[:parseval].max, tol)]
        gates + result['checks'].map { |name, value| name == 'fftshift' ? Gates.eq(name, value, 0) : Gates.le(name, value, tol) }
      end

      def bench_step(fixture, variant = nil)
        raise Error, "taylor_green has no variant #{variant}" unless variant.nil? || variant == 'rk4'

        spec = fixture.attr(:spec)
        omega = fixture.device(:omega_random)
        grid = grid(omega.shape[0], spec['nu'], spec['dt'])
        w = XFT.rfft2(omega)
        -> { step(w, grid) }
      end

      private

      def abs2(z) = (z.real * z.real) + (z.imag * z.imag)

      # Masks come from Ruby arrays, so no Bit array reaches the device.
      def grid(n, nu, dt)
        ky = XFT.fftfreq(n, d: 1.0 / n).reshape(n, 1)
        kx = XFT.rfftfreq(n, d: 1.0 / n).reshape(1, (n / 2) + 1)
        k2 = XF.cast((ky * ky) + (kx * kx))
        safe = k2.dup
        safe[0, 0] = 1
        inverse = 1 / safe
        inverse[0, 0] = 0
        keep = ->(ks) { XF.cast(ks.map { |k| k.abs < n / 3.0 ? 1 : 0 }) }
        ky_values = (0...n).map { |i| i < n / 2 ? i : i - n }
        mask = keep.call(ky_values).reshape(n, 1) * keep.call((0..(n / 2)).to_a).reshape(1, (n / 2) + 1)
        { n: n, dt: dt, ikx: XC.cast(kx) * Complex::I, iky: XC.cast(ky) * Complex::I, inverse: inverse, mask: mask,
          full: XM::NMath.exp(k2 * (-nu * dt)), half: XM::NMath.exp(k2 * (-nu * dt / 2)) }
      end

      # -div(u omega), with psi = omega / k^2, u = d psi / dy, v = -d psi / dx.
      def rhs(w, g)
        shape = [g[:n], g[:n]]
        psi = w * g[:inverse]
        u = XFT.irfft2(psi * g[:iky], s: shape)
        v = XFT.irfft2(-(psi * g[:ikx]), s: shape)
        omega = XFT.irfft2(w, s: shape)
        flux = (g[:ikx] * XFT.rfft2(u * omega)) + (g[:iky] * XFT.rfft2(v * omega))
        -(flux * g[:mask])
      end

      def step(w, g)
        dt = g[:dt]
        a = rhs(w, g) * dt
        b = rhs(g[:half] * (w + (a * 0.5)), g) * dt
        c = rhs((g[:half] * w) + (b * 0.5), g) * dt
        d = rhs((g[:full] * w) + (g[:half] * c), g) * dt
        (g[:full] * w) + (((g[:full] * a) + (2 * g[:half] * (b + c)) + d) / 6)
      end

      # Sums over the full spectrum from the half that rfft2 keeps: every
      # column but the first and the Nyquist one stands for two.
      def spectral_sum(z, n)
        (2 * abs2(z).sum) - abs2(z[true, 0]).sum - abs2(z[true, n / 2]).sum
      end

      # The sums stay on the device until the loop is over.
      def evolve(omega, g, steps)
        n = g[:n]
        w = XFT.rfft2(omega)
        energies = []
        enstrophies = []
        parseval = []
        (steps + 1).times do |i|
          w = step(w, g) if i.positive?
          real = XFT.irfft2(w, s: [n, n])
          z = spectral_sum(w, n)
          energies << spectral_sum(w * XM::NMath.sqrt(g[:inverse]), n)
          enstrophies << z
          parseval << [(real * real).sum * (n * n), z]
        end
        { energy: energies.map { |e| FFTBench.scalar(e) }, enstrophy: enstrophies.map { |e| FFTBench.scalar(e) },
          parseval: parseval.map { |r, z| (FFTBench.scalar(r) - FFTBench.scalar(z)).abs / FFTBench.scalar(z) } }
      end

      def drift(series) = (series.last - series.first).abs / series.first

      def values(taylor, random)
        { 'taylor.peak' => taylor[:peak], 'random.energy' => random[:energy].first, 'random.energy_end' => random[:energy].last,
          'random.enstrophy' => random[:enstrophy].first, 'random.enstrophy_end' => random[:enstrophy].last }
      end

      def checks(omega, n)
        spectrum = XFT.rfft2(omega)
        shifted = XFT.fftshift(XFT.fftfreq(n, d: 1.0 / n))
        { 'round_trip' => Gates.rel_max(XFT.irfft2(spectrum, s: [n, n]), omega),
          'transposed' => Gates.rel_max(XFT.rfft2(omega.transpose, axes: [-1, -2]).transpose, spectrum),
          'fftshift' => Gates.mismatches(shifted, XM::DFloat.new(n).seq - (n / 2)) }
      end
    end
  end
end
