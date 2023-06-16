from typing import Optional, Union, Tuple

import numpy as np
import casadi as ca


class Atmosphere1976:
    def __init__(self, use_metric: bool = True,
                 gravity: Optional[float] = None,
                 earth_radius: Optional[float] = None,
                 gas_constant: Optional[float] = None,
                 boundary_thickness: Optional[float] = None,
                 ):

        self.boundary_thickness = boundary_thickness

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

        if self.boundary_thickness is not None:
            h_sym = ca.SX.sym('h')
            temperature_expr, pressure_expr, density_expr = self.get_ca_atm_expr(h_sym, geometric=False)

            # Create 5th order polynomial to create atm model with continuous 2nd derivatives
            num_poly_coeffs = 6
            num_atm_layers = len(self.h_layers)
            num_boundary_layers = num_atm_layers - 1
            self.boundary_T_coefficients = np.empty((num_poly_coeffs, num_atm_layers + num_boundary_layers))
            self.boundary_T_coefficients[:] = np.nan
            self.boundary_P_coefficients = self.boundary_T_coefficients.copy()
            self.boundary_rho_coefficients = self.boundary_T_coefficients.copy()

            # indices where the boundary layers will be
            idces = np.arange(1, num_atm_layers + num_boundary_layers, 2)
            for idx in idces:  # skip last idx (no boundary)
                # Get altitude boundary
                altitude0 = self.h_layers[idx] - 0.5 * self.boundary_thickness
                altitude1 = self.h_layers[idx] + 0.5 * self.boundary_thickness
                coeffs_T, output_T = self.fit_boundary_layer(
                    temperature_expr, h_sym, altitude0, altitude1
                )
                coeffs_P, output_P = self.fit_boundary_layer(
                    pressure_expr, h_sym, altitude0, altitude1
                )
                coeffs_rho, output_rho = self.fit_boundary_layer(
                    density_expr, h_sym, altitude0, altitude1
                )

                # Insert boundary layer base at idx
                self.h_layers = np.insert(self.h_layers, idx, altitude0)
                self.lapse_layers = np.insert(self.lapse_layers, idx, np.nan)
                self.T_layers = np.insert(self.T_layers, idx, np.nan)
                self.P_layers = np.insert(self.P_layers, idx, np.nan)
                self.rho_layers = np.insert(self.rho_layers, idx, np.nan)
                self.boundary_T_coefficients[:, idx] = coeffs_T.flatten()
                self.boundary_P_coefficients[:, idx] = coeffs_P.flatten()
                self.boundary_rho_coefficients[:, idx] = coeffs_rho.flatten()
                self.layer_names.insert(
                    idx, self.layer_names[idx-1] + '-' + self.layer_names[idx] + ' Boundary'
                )

                # Modify boundary layer at next idx to shift up by boundary layer
                self.h_layers[idx + 1] = altitude1
                self.T_layers[idx + 1] = output_T[3]
                self.P_layers[idx + 1] = output_P[3]
                self.rho_layers[idx + 1] = output_rho[3]

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

    @staticmethod
    def boundary_layer(altitude, dh, coeffs_T, coeffs_P, coeffs_rho, altitude_0):
        # Fifth-order polynomial of the form:
        # p = C0 * [(h - h0)/dh]^0 + ... + C5 * [(h - h0)/dh]^5
        h_normalized = (altitude - altitude_0) / dh

        temperature = 0
        pressure = 0
        density = 0

        for idx, (coeff_T, coeff_P, coeff_rho) in enumerate(zip(coeffs_T, coeffs_P, coeffs_rho)):
            temperature += coeff_T * h_normalized ** idx
            pressure += coeff_P * h_normalized ** idx
            density += coeff_rho * h_normalized ** idx
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

        # (altitude, dh, coeffs_T, coeffs_P, coeffs_rho, altitude_0):

        if np.isnan(lapse_rate):
            return self.boundary_layer(
                altitude=altitude_geopotential,
                dh=self.boundary_thickness,
                coeffs_T=self.boundary_T_coefficients[:, layer_idx],
                coeffs_P=self.boundary_P_coefficients[:, layer_idx],
                coeffs_rho=self.boundary_rho_coefficients[:, layer_idx],
                altitude_0=self.h_layers[layer_idx]
            )
        elif lapse_rate == 0:
            return self.isothermal_layer(
                altitude=altitude_geopotential,
                altitude_0=self.h_layers[layer_idx],
                temperature_0=self.T_layers[layer_idx],
                pressure_0=self.P_layers[layer_idx],
                density_0=self.rho_layers[layer_idx]
            )
        else:
            return self.gradient_layer(
                altitude=altitude_geopotential,
                lapse_rate=lapse_rate,
                altitude_0=self.h_layers[layer_idx],
                temperature_0=self.T_layers[layer_idx],
                pressure_0=self.P_layers[layer_idx],
                density_0=self.rho_layers[layer_idx]
            )

    def temperature(self, altitude_geometric):
        temperature, _, __ = self.atm_data(altitude_geometric)
        return temperature

    def pressure(self, altitude_geometric):
        _, pressure, __ = self.atm_data(altitude_geometric)

    def density(self, altitude_geometric):
        _, __, density = self.atm_data(altitude_geometric)
        return density

    def speed_of_sound(self, altitude_geometric):
        temperature = self.temperature(altitude_geometric)
        speed_of_sound = np.sqrt(self.specific_heat_ratio * self.gas_constant * temperature)
        return speed_of_sound

    def layer(self, altitude_geometric):
        altitude_geopotential = self.geometric2geopotential(altitude_geometric)
        if altitude_geopotential < self.h_layers[0]:
            altitude_geopotential = self.h_layers[0]
        layer_idx = np.sum(altitude_geopotential >= self.h_layers) - 1

        return self.layer_names[layer_idx]

    def get_ca_atm_expr(
            self, altitude: Union[ca.SX, ca.MX], geometric: bool = True) -> Tuple[Union[ca.SX, ca.MX],
                                                                                  Union[ca.SX, ca.MX],
                                                                                  Union[ca.SX, ca.MX]]:

        type_h = type(altitude)  # SX or MX

        # If geometric, convert to geopotential
        if geometric:
            altitude_geopotential = self.geometric2geopotential(altitude)
        else:
            altitude_geopotential = altitude
        layer_idx = ca.sum1(altitude_geopotential >= self.h_layers) - 1

        temperature = type_h(0)
        pressure = type_h(0)
        density = type_h(0)
        for idx, lapse_rate in enumerate(self.lapse_layers):
            if np.isnan(lapse_rate):
                temperature_i, pressure_i, density_i = self.boundary_layer(
                    altitude=altitude_geopotential,
                    dh=self.boundary_thickness,
                    coeffs_T=self.boundary_T_coefficients[:, idx],
                    coeffs_P=self.boundary_P_coefficients[:, idx],
                    coeffs_rho=self.boundary_rho_coefficients[:, idx],
                    altitude_0=self.h_layers[idx]
                )
            elif lapse_rate == 0:
                temperature_i, pressure_i, density_i = self.isothermal_layer(
                    altitude=altitude_geopotential,
                    altitude_0=self.h_layers[idx],
                    temperature_0=self.T_layers[idx],
                    pressure_0=self.P_layers[idx],
                    density_0=self.rho_layers[idx]
                )
            else:
                temperature_i, pressure_i, density_i = self.gradient_layer(
                    altitude=altitude_geopotential,
                    lapse_rate=lapse_rate,
                    altitude_0=self.h_layers[idx],
                    temperature_0=self.T_layers[idx],
                    pressure_0=self.P_layers[idx],
                    density_0=self.rho_layers[idx]
                )
            temperature += ca.if_else(layer_idx == idx, temperature_i, 0)
            pressure += ca.if_else(layer_idx == idx, pressure_i, 0)
            density += ca.if_else(layer_idx == idx, density_i, 0)

        return temperature, pressure, density

    def get_ca_speed_of_sound_expr(self, altitude_geometric: Union[ca.SX, ca.MX]):
        temperature, _, __ = self.get_ca_atm_expr(altitude_geometric)
        speed_of_sound = np.sqrt(self.specific_heat_ratio * self.gas_constant * temperature)
        return speed_of_sound

    @staticmethod
    def fit_boundary_layer(f: ca.SX, h: ca.SX, h0: float, h1: float):
        # Fit coefficients for a fifth-order polynomial of the form:
        # p = C0 * [(h - h0)/dh]^0 + ... + C5 * [(h - h0)/dh]^5
        # Return C = [C0, C1, ..., C5]
        fh = ca.jacobian(f, h)
        fhh = ca.jacobian(fh, h)
        f_fun = ca.Function('f', (h,), (f,))
        fh_fun = ca.Function('fh', (h,), (fh,))
        fhh_fun = ca.Function('fhh', (h,), (fhh,))

        dh = h1 - h0
        dh2 = dh**2

        output = np.asarray((
            f_fun(h0), fh_fun(h0), fhh_fun(h0),
            f_fun(h1), fh_fun(h1), fhh_fun(h1)
        )).reshape((-1, 1))

        design_matrix = np.array((
            (1, 0, 0, 0, 0, 0),
            (0, 1/dh, 0, 0, 0, 0),
            (0, 0, 2/dh2, 0, 0, 0),
            (1, 1, 1, 1, 1, 1),
            (0, 1/dh, 2/dh, 3/dh, 4/dh, 5/dh),
            (0, 0, 2/dh2, 6/dh2, 12/dh2, 20/dh2)
        ))

        coefficients = np.linalg.solve(design_matrix, output)

        # TODO - remove (validation)
        x0 = 0
        x1 = 1
        f0 = 0
        f1 = 0
        for idx, coeff in enumerate(coefficients):
            f0 += coeff * x0 ** idx
            f1 += coeff * x1 ** idx

        return coefficients, output


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

    atm = Atmosphere1976(use_metric=False, earth_radius=re, gravity=g0, boundary_thickness=1000)
    density_1976 = np.empty(shape=altitudes.shape)
    layer_1976 = list()

    h_sx = ca.SX.sym('h')
    _, __, rho_expr = atm.get_ca_atm_expr(h_sx)

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

    ax21.plot(density_exponential, altitudes / 10_000, label='Exponential Atm.')
    ax22.plot(-density_deriv_exponential, altitudes / 10_000, label='Exponential Atm.')
    for layer in atm.layer_names:
        layer_idcs = np.where(layer_1976 == layer)
        if len(layer_idcs[0]) > 0:
            ax21.plot(density_1976_ca[layer_idcs], altitudes[layer_idcs] / 10_000, '--', label=layer)
            ax22.plot(-density_1976_deriv_ca[layer_idcs], altitudes[layer_idcs] / 10_000, '--', label=layer)
    xlim21 = ax21.get_xlim()
    xlim22 = ax22.get_xlim()
    ax21.plot(xlim21, np.array((1, 1)) * 80_000 / 10_000, 'k--', zorder=0)
    ax22.plot(xlim22, np.array((1, 1)) * 80_000 / 10_000, 'k--', zorder=0)
    ax21.set_xscale('log')
    ax22.set_xscale('log')
    ax21.grid()
    ax22.grid()

    ax21.set_ylabel('Altitude [10,000 ft]')
    ax21.set_xlabel(r'$\rho$ [slug / ft$^3$]')
    ax22.set_xlabel(r'$-\dfrac{d\rho}{dh}$ [slug / ft$^4$]')
    # ax21.set_xscale('log')

    ax21.legend(loc='upper right')

    fig2.tight_layout()

    fig2.savefig('atm_comparison.eps',
                 format='eps',
                 bbox_inches='tight')

    plt.show()
