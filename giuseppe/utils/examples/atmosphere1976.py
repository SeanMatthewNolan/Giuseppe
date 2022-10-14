from typing import Optional, Union

import numpy as np
import casadi as ca


class Atmosphere1976:
    def __init__(self, use_metric: bool = True,
                 gravity: Optional[float] = None,
                 earth_radius: Optional[float] = None,
                 gas_constant: Optional[float] = None
                 ):

        self.h_layers = np.array((-610, 11_000, 20_000, 32_000, 47_000, 51_000, 71_000, 84_852))  # m
        self.lapse_layers = np.array((-6.5, 0, 1, 2.8, 0, -2.8, -2, 0)) * 1e-3  # K/m
        self.T_layers = np.array((292.15,))  # K
        self.P_layers = np.array((108_900,))  # Pa
        self.rho_layers = np.array((1.2985,))  # kg/m^3
        self.layer_names = ['Troposphere', 'Tropopause',
                            'Lower Stratosphere', 'Upper Stratosphere', 'Stratopause',
                            'Lower Mesosphere', 'Upper Mesosphere', 'Mesopause']

        if gas_constant is not None:
            self.gas_constant = gas_constant
        else:
            self.gas_constant = 287.05  # J/kg-K = N-m/kg-K

        if earth_radius is not None:
            self.earth_radius = earth_radius
        else:
            self.earth_radius = 6_371_000

        if gravity is not None:
            self.gravity = gravity
        else:
            self.gravity = 9.807

        if not use_metric:
            slug2kg = 14.5939
            kg2slug = 1 / slug2kg
            m2ft = 3.28084
            ft2m = 1 / m2ft
            kelvin2rankine = 1.8
            rankine2kelvin = 1 / kelvin2rankine
            pascal2psf = 0.020885
            newtons2lb = 0.2248090795

            if gas_constant is None:
                # N-m/kg K * (lb/N) * (ft/m) * (kg/slug) * (K/R) = ft-lb / slug - R
                self.gas_constant = self.gas_constant * newtons2lb * m2ft * slug2kg * rankine2kelvin
            if earth_radius is None:
                self.earth_radius = self.earth_radius * m2ft  # m * (ft/m) = ft
            if gravity is None:
                self.gravity = self.gravity * m2ft  # m/s**2 * (ft/m) = ft/s**2

            self.h_layers = self.h_layers * 3.28084  # m * (ft/m) = ft
            self.lapse_layers = self.lapse_layers * ft2m * kelvin2rankine  # K/m * (m/ft) * (R/K) = R/ft
            self.T_layers = self.T_layers * kelvin2rankine  # K * (R/K) = R
            self.P_layers = self.P_layers * pascal2psf  # Pa * (psf / Pa) = psf = lb/ft**2
            self.rho_layers = self.rho_layers * kg2slug * ft2m ** 3  # kg/m**3 * (slug/kg) * (m/ft)**3

        self.specific_heat_ratio = 1.4
        self.build_layers()

    def isothermal_layer(self, altitude, altitude_0, temperature_0, pressure_0, density_0):
        temperature = temperature_0
        exponential = np.exp(-self.gravity / (self.gas_constant * temperature) * (altitude - altitude_0))
        pressure = pressure_0 * exponential
        density = density_0 * exponential
        return temperature, pressure, density

    def gradient_layer(self, altitude, lapse_rate, altitude_0, temperature_0, pressure_0, density_0):
        temperature = temperature_0 + lapse_rate * (altitude - altitude_0)
        pressure = pressure_0 * (temperature / temperature_0) ** (-self.gravity / (lapse_rate * self.gas_constant))
        density = density_0 * (temperature / temperature_0) ** (-1 - self.gravity / (lapse_rate * self.gas_constant))
        return temperature, pressure, density

    def build_layers(self):
        for idx, lapse_rate in enumerate(self.lapse_layers[:-1]):
            if lapse_rate == 0:  # Isothermal Layer
                _temperature, _pressure, _density = self.isothermal_layer(altitude=self.h_layers[idx + 1],
                                                                          altitude_0=self.h_layers[idx],
                                                                          temperature_0=self.T_layers[idx],
                                                                          pressure_0=self.P_layers[idx],
                                                                          density_0=self.rho_layers[idx])
            else:
                _temperature, _pressure, _density = self.gradient_layer(altitude=self.h_layers[idx + 1],
                                                                        lapse_rate=lapse_rate,
                                                                        altitude_0=self.h_layers[idx],
                                                                        temperature_0=self.T_layers[idx],
                                                                        pressure_0=self.P_layers[idx],
                                                                        density_0=self.rho_layers[idx])
            self.T_layers = np.append(self.T_layers, _temperature)
            self.P_layers = np.append(self.P_layers, _pressure)
            self.rho_layers = np.append(self.rho_layers, _density)

    def geometric2geopotential(self, h_geometric):
        return self.earth_radius * h_geometric / (self.earth_radius + h_geometric)

    def geopotential2geometric(self, h_geopotential):
        return self.earth_radius * h_geopotential / (self.earth_radius - h_geopotential)

    def atm_data(self, altitude_geometric):
        altitude_geopotential = self.geometric2geopotential(altitude_geometric)
        if altitude_geopotential < self.h_layers[0]:
            altitude_geopotential = self.h_layers[0]

        layer_idx = np.sum(altitude_geopotential >= self.h_layers) - 1
        lapse_rate = self.lapse_layers[layer_idx]
        if lapse_rate == 0:
            return self.isothermal_layer(altitude=altitude_geopotential,
                                         altitude_0=self.h_layers[layer_idx],
                                         temperature_0=self.T_layers[layer_idx],
                                         pressure_0=self.P_layers[layer_idx],
                                         density_0=self.rho_layers[layer_idx])
        else:
            return self.gradient_layer(altitude=altitude_geopotential,
                                       lapse_rate=lapse_rate,
                                       altitude_0=self.h_layers[layer_idx],
                                       temperature_0=self.T_layers[layer_idx],
                                       pressure_0=self.P_layers[layer_idx],
                                       density_0=self.rho_layers[layer_idx])

    def temperature(self, altitude_geometric):
        temperature, _, __ = self.atm_data(altitude_geometric)
        return temperature

    def pressure(self, altitude_geometric):
        _, pressure, __ = self.atm_data(altitude_geometric)

    def density(self, altitude_geometric):
        _, __, density = self.atm_data(altitude_geometric)
        return density

    def layer(self, altitude_geometric):
        altitude_geopotential = self.geometric2geopotential(altitude_geometric)
        if altitude_geopotential < self.h_layers[0]:
            altitude_geopotential = self.h_layers[0]
        layer_idx = np.sum(altitude_geopotential >= self.h_layers) - 1

        return self.layer_names[layer_idx]

    def get_sx_atm_expr(self, altitude_geometric: Union[ca.SX, ca.MX]):
        altitude_geopotential = self.geometric2geopotential(altitude_geometric)
        layer_idx = ca.sum1(altitude_geopotential >= self.h_layers) - 1

        temperature = 0
        pressure = 0
        density = 0
        for idx, lapse_rate in enumerate(self.lapse_layers):
            if lapse_rate == 0:
                temperature_i, pressure_i, density_i = self.isothermal_layer(altitude=altitude_geopotential,
                                                                             altitude_0=self.h_layers[idx],
                                                                             temperature_0=self.T_layers[idx],
                                                                             pressure_0=self.P_layers[idx],
                                                                             density_0=self.rho_layers[idx])
            else:
                temperature_i, pressure_i, density_i = self.gradient_layer(altitude=altitude_geopotential,
                                                                           lapse_rate=lapse_rate,
                                                                           altitude_0=self.h_layers[idx],
                                                                           temperature_0=self.T_layers[idx],
                                                                           pressure_0=self.P_layers[idx],
                                                                           density_0=self.rho_layers[idx])
            temperature += ca.if_else(layer_idx == idx, temperature_i, 0)
            pressure += ca.if_else(layer_idx == idx, pressure_i, 0)
            density += ca.if_else(layer_idx == idx, density_i, 0)

        return temperature, pressure, density


# TODO refactor callback into backend
class CasidiFunction(ca.Callback):
    def __init__(self, eval_func, func_name: str = 'func', n_in: int = 1, n_out: int = 1,
                 options: Optional[dict] = None):
        ca.Callback.__init__(self)
        self.eval_func = eval_func
        self.n_in = n_in
        self.n_out = n_out

        if options is None:
            options = {"enable_fd": True, "fd_method": "smoothing"}

        self.construct(func_name, options)

    def get_n_in(self):
        return self.n_in

    def get_n_out(self):
        return self.n_out

    @staticmethod
    def init(): return

    def eval(self, args):
        return self.eval_func(*args)


if __name__ == "__main__":
    from matplotlib import pyplot as plt

    altitudes = np.linspace(0, 260_000, 10_000)

    rho_0 = 0.002378  # slug/ft**3
    h_ref = 23_800  # ft

    density_exponential = rho_0 * np.exp(-altitudes / h_ref)
    density_deriv_exponential = -rho_0 / h_ref * np.exp(-altitudes / h_ref)

    re = 20_902_900
    mu = 0.14076539e17
    g0 = mu / re**2

    atm = Atmosphere1976(use_metric=False, earth_radius=re, gravity=g0)
    density_1976 = np.empty(shape=altitudes.shape)
    layer_1976 = list()

    h_sx = ca.SX.sym('h')
    _, __, rho_expr = atm.get_sx_atm_expr(h_sx)

    ca_rho_func = ca.Function('rho', (h_sx,), (rho_expr,), ('h',), ('rho',))
    rho_expr_deriv = ca.jacobian(rho_expr, h_sx)
    ca_rho_deriv_func = ca.Function('drho_dh', (h_sx,), (rho_expr_deriv,), ('h',), ('drho_dh',))

    density_1976_ca = np.empty(shape=altitudes.shape)
    density_1976_deriv_ca = np.empty(shape=altitudes.shape)

    for i, h in enumerate(altitudes):
        density_1976[i] = atm.density(h)
        layer_1976.append(atm.layer(h))

        density_1976_ca[i] = float(ca_rho_func(h))
        density_1976_deriv_ca[i] = float(ca_rho_deriv_func(h))

    layer_1976 = np.array(layer_1976)

    fig1 = plt.figure(figsize=(6.5, 5))
    title = fig1.suptitle('Compare 1976 to Exponential Atmosphere')

    ax11 = fig1.add_subplot(111)
    ax11.plot(density_exponential * 100_000, altitudes / 10_000, label='Exponential')
    for layer in atm.layer_names:
        layer_idcs = np.where(layer_1976 == layer)
        if len(layer_idcs[0]) > 0:
            ax11.plot(density_1976[layer_idcs] * 100_000, altitudes[layer_idcs] / 10_000, label=layer + ' 1976')
    xlim11 = ax11.get_xlim()
    ax11.plot(xlim11, np.array((1, 1)) * 80_000 / 10_000, 'k--', zorder=0)
    ax11.set_xlim(xlim11)
    ax11.grid()
    ax11.set_xlabel('Density [slug / 100,000 ft^3]')
    ax11.set_ylabel('Altitude [10,000 ft]')
    ax11.legend()

    fig2 = plt.figure(figsize=(6.5, 5))
    # title = fig2.suptitle('Comparison of Exponential and 1976 Standard Atmosphere')

    ax21 = fig2.add_subplot(121)
    ax22 = fig2.add_subplot(122)

    ax21.plot(density_exponential * 100_000, altitudes / 10_000, label='Exponential Atm.')
    ax22.plot(density_deriv_exponential * 1e9, altitudes / 10_000, label='Exponential Atm.')
    for layer in atm.layer_names:
        layer_idcs = np.where(layer_1976 == layer)
        if len(layer_idcs[0]) > 0:
            ax21.plot(density_1976_ca[layer_idcs] * 100_000, altitudes[layer_idcs] / 10_000, '--', label=layer)
            ax22.plot(density_1976_deriv_ca[layer_idcs] * 1e9, altitudes[layer_idcs] / 10_000, '--', label=layer)
    xlim21 = ax21.get_xlim()
    xlim22 = ax22.get_xlim()
    ax21.plot(xlim21, np.array((1, 1)) * 80_000 / 10_000, 'k--', zorder=0)
    ax22.plot(xlim22, np.array((1, 1)) * 80_000 / 10_000, 'k--', zorder=0)
    ax21.grid()
    ax22.grid()

    ax21.set_ylabel('Altitude [10,000 ft]')
    ax21.set_xlabel(r'$\rho$ [slug / 100,000 ft$^3$]')
    ax22.set_xlabel(r'$-\dfrac{d\rho}{dh}$ [slug / 10$^9$ ft$^4$]')
    # ax21.set_xscale('log')

    ax21.legend()

    fig2.tight_layout()

    fig2.savefig('atm_comparison.eps',
                 format='eps',
                 bbox_inches='tight')

    plt.show()
