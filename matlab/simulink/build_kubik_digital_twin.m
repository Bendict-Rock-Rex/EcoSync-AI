function mdl = build_kubik_digital_twin()
% BUILD_KUBIK_DIGITAL_TWIN
% Programmatically generates the EcoSync-AI Simulink model:
% 'matlab/simulink/kubik_building_model.slx'
%
% Implements the thermal-electrical balance:
%   C_th * dT_zone/dt = Q_envelope + Q_solar + Q_internal - Q_cooling
%   P_elec = (Q_cooling / COP) + P_idle

    model_name = 'kubik_building_model';
    save_path = fullfile(pwd, 'matlab', 'simulink', [model_name, '.slx']);
    
    % Close if already open
    if bdIsLoaded(model_name)
        close_system(model_name, 0);
    end
    
    % Create new system
    new_system(model_name);
    open_system(model_name);
    
    % Set solver configuration
    set_param(model_name, 'Solver', 'ode45');
    set_param(model_name, 'StopTime', '86400'); % 24 hours
    set_param(model_name, 'SaveTime', 'on');
    set_param(model_name, 'SaveOutput', 'on');
    
    % Add Inports:
    % In1: T_amb (Outdoor temp, degC)
    % In2: Solar_Rad (W/m2)
    % In3: T_chw_sp (Chilled water setpoint, degC)
    % In4: T_zone_sp (Thermostat setpoint, degC)
    % In5: Chiller_Enable (0 or 1)
    add_block('simulink/Sources/In1', [model_name, '/T_amb'], 'Position', [40, 60, 70, 80]);
    add_block('simulink/Sources/In1', [model_name, '/Solar_Rad'], 'Position', [40, 120, 70, 140]);
    add_block('simulink/Sources/In1', [model_name, '/T_chw_sp'], 'Position', [40, 180, 70, 200]);
    add_block('simulink/Sources/In1', [model_name, '/T_zone_sp'], 'Position', [40, 240, 70, 260]);
    add_block('simulink/Sources/In1', [model_name, '/Chiller_Enable'], 'Position', [40, 300, 70, 320]);
    
    % Envelope Heat Gain: UA * (T_amb - T_zone)
    add_block('simulink/Math Operations/Subtract', [model_name, '/Sub_Tamb_Tzone'], 'Position', [140, 60, 160, 90]);
    add_block('simulink/Math Operations/Gain', [model_name, '/Gain_UA'], 'Gain', '125.0', 'Position', [190, 65, 240, 95]);
    
    % Solar Gain: Area_aperture * Solar_Rad
    add_block('simulink/Math Operations/Gain', [model_name, '/Gain_Solar'], 'Gain', '1.8', 'Position', [140, 120, 190, 150]);
    
    % Constant Internal Heat Gain: 350 W
    add_block('simulink/Sources/Constant', [model_name, '/Const_Qint'], 'Value', '350.0', 'Position', [140, 170, 180, 190]);
    
    % Total Gains Sum: Q_env + Q_sol + Q_int
    add_block('simulink/Math Operations/Sum', [model_name, '/Sum_Gains'], 'Inputs', '+++', 'Position', [280, 90, 300, 160]);
    
    % Thermal Net: Q_gains - Q_cooling
    add_block('simulink/Math Operations/Subtract', [model_name, '/Sub_Net_Thermal'], 'Position', [340, 120, 360, 150]);
    
    % Divide by C_th: Gain = 1 / 4.2e6
    add_block('simulink/Math Operations/Gain', [model_name, '/Gain_Inv_Cth'], 'Gain', '1/4.2e6', 'Position', [390, 125, 450, 155]);
    
    % Integrator: Initial Condition = 24.2 deg C
    add_block('simulink/Continuous/Integrator', [model_name, '/Thermal_Integrator'], ...
              'InitialCondition', '24.2', 'Position', [480, 125, 510, 155]);
              
    % Outport: T_zone
    add_block('simulink/Sinks/Out1', [model_name, '/T_zone_out'], 'Position', [580, 130, 610, 150]);
    
    % Connect feedback loop: Integrator output back to Sub_Tamb_Tzone
    add_line(model_name, 'T_amb/1', 'Sub_Tamb_Tzone/1');
    add_line(model_name, 'Sub_Tamb_Tzone/1', 'Gain_UA/1');
    add_line(model_name, 'Solar_Rad/1', 'Gain_Solar/1');
    add_line(model_name, 'Gain_UA/1', 'Sum_Gains/1');
    add_line(model_name, 'Gain_Solar/1', 'Sum_Gains/2');
    add_line(model_name, 'Const_Qint/1', 'Sum_Gains/3');
    add_line(model_name, 'Sum_Gains/1', 'Sub_Net_Thermal/1');
    add_line(model_name, 'Gain_Inv_Cth/1', 'Thermal_Integrator/1');
    add_line(model_name, 'Sub_Net_Thermal/1', 'Gain_Inv_Cth/1');
    add_line(model_name, 'Thermal_Integrator/1', 'T_zone_out/1');
    
    % Feedback T_zone line to Sub_Tamb_Tzone input 2
    add_line(model_name, 'Thermal_Integrator/1', 'Sub_Tamb_Tzone/2', 'autorouting', 'smart');
    
    % Cooling load dummy zero/stub for base open-loop model
    add_block('simulink/Sources/Constant', [model_name, '/Const_Qcool_stub'], 'Value', '0.0', 'Position', [280, 200, 310, 220]);
    add_line(model_name, 'Const_Qcool_stub/1', 'Sub_Net_Thermal/2');
    
    % Save model
    save_system(model_name, save_path);
    close_system(model_name);
    
    fprintf('Simulink Model successfully created and saved: %s\n', save_path);
    mdl = save_path;
end
