function results = run_building_simulation(inputs_file, outputs_file, params_file)
% RUN_BUILDING_SIMULATION
% Executes the EcoSync-AI Building Digital Twin ODE / State-Space Simulation
%
% Inputs:
%   inputs_file  - Path to JSON containing:
%                  - 'time_sec': array of timestamps (e.g. 0:600:86400)
%                  - 'T_amb': outdoor dry-bulb temperature [degC]
%                  - 'solar_rad': solar radiation [W/m2]
%                  - 'chiller_on': chiller binary order [0 or 1]
%                  - 'T_chw_sp': chilled water supply setpoint [degC]
%                  - 'T_zone_sp': zone thermostat setpoint [degC]
%                  - 'initial_T_zone': starting zone temperature [degC]
%   outputs_file - Path to output JSON for simulation results
%   params_file  - (Optional) Path to MATLAB parameter file or struct
%
% Output:
%   results struct written to outputs_file containing:
%   - time_sec, T_zone, chiller_power_w, COP, Q_cooling_w, power_factor, comfort_violation

    if nargin < 1 || isempty(inputs_file)
        inputs_file = 'matlab/simulation_inputs.json';
    end
    if nargin < 2 || isempty(outputs_file)
        outputs_file = 'matlab/results/simulation_results.json';
    end
    
    % Load default physical parameters
    run('matlab/simulink_parameters.m'); % defines 'p'
    
    % Read input JSON
    if ~isfile(inputs_file)
        error('Input file not found: %s', inputs_file);
    end
    raw_str = fileread(inputs_file);
    sim_in = jsondecode(raw_str);
    
    time_sec = sim_in.time_sec(:);
    N = length(time_sec);
    T_amb = sim_in.T_amb(:);
    solar_rad = sim_in.solar_rad(:);
    chiller_on = sim_in.chiller_on(:);
    T_chw_sp = sim_in.T_chw_sp(:);
    T_zone_sp = sim_in.T_zone_sp(:);
    
    if isfield(sim_in, 'initial_T_zone') && ~isempty(sim_in.initial_T_zone)
        T_z0 = sim_in.initial_T_zone;
    else
        T_z0 = p.T_zone_initial;
    end
    
    % Preallocate outputs
    T_zone = zeros(N, 1);
    Q_cooling = zeros(N, 1);
    P_elec = zeros(N, 1);
    COP_eff = zeros(N, 1);
    PF_eff = zeros(N, 1);
    comfort_violation = zeros(N, 1);
    
    T_curr = T_z0;
    dt = 600.0; % default 10-minute step
    
    for k = 1:N
        if k > 1
            dt = time_sec(k) - time_sec(k-1);
            if dt <= 0, dt = 600.0; end
        end
        
        % 1. Thermal loads
        Q_env = p.UA * (T_amb(k) - T_curr);
        Q_sol = p.solar_aperture * solar_rad(k);
        Q_int = p.q_internal;
        Q_gain = Q_env + Q_sol + Q_int;
        
        % 2. HVAC Cooling Extraction
        % If chiller is commanded ON and zone temperature is above thermostat setpoint
        if chiller_on(k) > 0.5 && T_curr > T_chw_sp(k)
            % Maximum cooling capacity from fan-coil heat exchanger:
            % Q_fc = UA_fc * (T_zone - T_chw)
            Q_avail = p.UA_fc * max(0.0, T_curr - T_chw_sp(k));
            
            % Thermostatic throttling to track zone setpoint:
            if T_curr > T_zone_sp(k)
                % Proportional demand with capacity saturation
                gain_prop = 1200.0; % W / K
                Q_demand = gain_prop * (T_curr - T_zone_sp(k));
                Q_cool = min(Q_avail, max(0.0, Q_demand));
            else
                Q_cool = 0.0;
            end
        else
            Q_cool = 0.0;
        end
        
        % 3. Discrete-time thermal integration: C_th * dT/dt = Q_gain - Q_cool
        dT = ((Q_gain - Q_cool) / p.C_th) * dt;
        T_curr = T_curr + dT;
        
        % 4. Chiller electrical power & COP computation
        if chiller_on(k) > 0.5
            % Lift-adjusted COP
            cop = p.COP_nominal * (1.0 + p.beta_chw * (T_chw_sp(k) - p.T_chw_nominal) ...
                                       - p.beta_amb * (T_amb(k) - 25.0));
            cop = max(1.2, min(5.5, cop)); % physics boundary
            
            if Q_cool > 50.0
                p_active = (Q_cool / cop) + p.P_chiller_idle;
                pf = p.pf_nominal;
            else
                p_active = p.P_chiller_idle;
                pf = p.pf_idle;
            end
            p_active = min(p.P_chiller_max, p_active);
        else
            cop = 0.0;
            p_active = 0.0;
            pf = 1.0;
        end
        
        % 5. Record step metrics
        T_zone(k) = T_curr;
        Q_cooling(k) = Q_cool;
        P_elec(k) = p_active;
        COP_eff(k) = cop;
        PF_eff(k) = pf;
        comfort_violation(k) = (T_curr > p.T_zone_max) || (T_curr < p.T_zone_min);
    end
    
    % Package results struct
    results = struct();
    results.time_sec = time_sec;
    results.T_zone_degc = round(T_zone, 3);
    results.chiller_power_w = round(P_elec, 1);
    results.cooling_thermal_w = round(Q_cooling, 1);
    results.cop = round(COP_eff, 3);
    results.power_factor = round(PF_eff, 3);
    results.comfort_violation = comfort_violation;
    results.total_energy_kwh = round(sum(P_elec * (dt / 3600)) / 1000.0, 3);
    results.peak_power_w = round(max(P_elec), 1);
    results.mean_zone_temp = round(mean(T_zone), 2);
    results.max_zone_temp = round(max(T_zone), 2);
    results.comfort_violation_count = sum(comfort_violation);
    
    % Save to output JSON
    out_dir = fileparts(outputs_file);
    if ~isempty(out_dir) && ~isfolder(out_dir)
        mkdir(out_dir);
    end
    out_json = jsonencode(results);
    fid = fopen(outputs_file, 'w');
    if fid == -1
        error('Cannot open file for writing: %s', outputs_file);
    end
    fprintf(fid, '%s', out_json);
    fclose(fid);
    
    fprintf('Simulation completed successfully. Total Energy: %.2f kWh, Peak: %.1f W\n', ...
        results.total_energy_kwh, results.peak_power_w);
end
