from typing import Optional

import numpy as np


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
                # N-m/kg K * (lb/N) * (ft/m) * (kg/slug) * (K/R) = ft-lb / slug - K
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

    def atm_func(self, altitude_geometric):
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


if __name__ == "__main__":
    from matplotlib import pyplot as plt

    altitudes = np.linspace(80_000, 260_000, 1_000)

    rho_0 = 0.002378  # slug/ft**3
    h_ref = 23_800  # ft

    density_exponential = rho_0 * np.exp(-altitudes / h_ref)

    re = 20_902_900
    mu = 0.14076539e17
    g0 = mu / re**2

    atm = Atmosphere1976(use_metric=False, earth_radius=re, gravity=g0)
    density_1976 = np.empty(shape=altitudes.shape)
    for i, h in enumerate(altitudes):
        _, __, density_1976[i] = atm.atm_func(h)

    fig = plt.figure(figsize=(6.5, 5))
    title = fig.suptitle('Compare 1976 to Exponential Atmosphere')

    ax1 = fig.add_subplot(111)
    ax1.plot(density_exponential, altitudes / 10_000, label='Exponential')
    ax1.plot(density_1976, altitudes / 10_000, label='1976')
    ax1.grid()
    ax1.set_xlabel('Density [slug / ft^3]')
    ax1.set_ylabel('Altitude [10,000 ft]')
    ax1.legend()

    plt.show()
