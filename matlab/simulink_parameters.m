%% EcoSync-AI Digital Twin Simulation Parameters
% Calibrated against KUBIK experimental building cooling dataset (July-October 2020)
% Reference: Zenodo 4746415 (HOLISDER / KUBIK experimental building, TECNALIA)

p = struct();

%% 1. Thermal Envelope Parameters (Calibrated 1st-order lumped zone model)
% Zone volume: ~120 m3, air density: 1.2 kg/m3, air heat capacity: 1005 J/kg-K
% Effective thermal mass includes interior structural slab and partitions:
% C_th in J/K (Watt-seconds per Kelvin)
p.C_th = 4.2e6;          % 4.2 MJ/K effective lumped zone heat capacity

% Overall envelope thermal conductance (UA value):
% Conductance to outdoor ambient: U*A in W/K
p.UA = 125.0;            % 125 W/K (high performance testbed insulation with fenestration)

% Solar aperture and internal gain coefficients:
p.solar_aperture = 1.8;  % m^2 effective solar aperture area (SHGC * Area)
p.q_internal = 350.0;    % W constant baseline internal gains (equipment/lighting)

%% 2. HVAC & Hydronic Cooling Parameters
% Fan-coil heat transfer coefficient (UA_fc in W/K):
p.UA_fc = 620.0;         % W/K fan-coil heat exchanger capacity

% Chilled water supply baseline:
p.T_chw_nominal = 10.5;  % deg C nominal supply setpoint
p.T_zone_initial = 24.2; % deg C initial zone temperature

% Chiller Compressor Electrical Power Model:
% Derived from physical COP vs chilled water lift:
% COP = COP_nominal * (1 + beta_chw * (T_chw - T_chw_nominal) - beta_amb * (T_amb - 25.0))
p.COP_nominal = 3.20;    % Nominal COP at 10.5 deg C supply and 25 deg C ambient
p.beta_chw = 0.032;      % +3.2% COP improvement per +1.0 deg C chiller setpoint float
p.beta_amb = 0.018;      % -1.8% COP degradation per +1.0 deg C outdoor ambient rise
p.P_chiller_idle = 180.0;% W auxiliary pump & standby active power
p.P_chiller_max = 3500.0;% W electrical maximum safety power bound

%% 3. Power Quality (Electrical Coupling)
p.nominal_voltage_ll = 400.0; % V nominal line-to-line 3-phase voltage
p.pf_nominal = 0.88;          % Operating power factor under nominal compressor load
p.pf_idle = 0.22;             % Standby power factor (inductive magnetizing current)

%% 4. Control Limits & Bounds
p.T_zone_min = 21.0;     % deg C comfort lower limit
p.T_zone_max = 26.0;     % deg C comfort upper limit
p.T_chw_min = 7.0;       % deg C chilled water lower limit
p.T_chw_max = 14.0;      % deg C chilled water upper limit
p.sample_time_sec = 600; % 10 minutes (600s) matching KUBIK sampling interval
