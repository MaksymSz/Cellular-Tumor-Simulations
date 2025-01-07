import PySimpleGUI as sg

def get_simulation_params():
    """
    Open a PySimpleGUI window that allows the user to:
      1. Choose a simulation type
      2. Edit parameters for that simulation
      3. Submit the parameters

    The function returns a dictionary with the parameters (and simulation type)
    once the user clicks 'Submit', or returns None if the window is closed/canceled.
    """

    # Default parameter dictionaries for each simulation type
    param_dict_1 = {
        'K_CC': 3,
        'K_HH': 0.5,
        'K_NN': 3,
        'I_nut': 400.0,
        'k': 7
    }

    param_dict_2 = {
        'DELTA_T': 24,
        'CCT': 1,
        'P_P': 0.0417,
        'MU': 10,
        'INIT_CELL': 'p_rtc_init',
        'P_A': 0,
        'P_S': 0,
        'P_MAX': 10
    }

    param_dict_3 = {
        'total_steps': 250,
        'initial_population': 0,
        'inject_population': 10,
        'division_rate_high_sial': 0.4,
        'division_rate_low_sial': 0.2,
        'death_rate': 0.02,
        'fusion_rate': 0.005,
        'suppressive_effect': 0.7,
        'permissive_effect': 1.8,
        'neutral_effect': 1.0,
        'mutation_rate': 0.01
    }

    # Helper to make a row with label on the left and input on the right
    def labeled_input(label, default_val, key):
        return [
            sg.Text(label),
            sg.Push(),  # "Push" expands the space between the Text and the Input
            sg.Input(str(default_val), key=key, size=(10, 1))
        ]

    # Column layouts for each parameter set
    col_layout_1 = [
        labeled_input('K_CC',  param_dict_1['K_CC'],  'K_CC'),
        labeled_input('K_HH',  param_dict_1['K_HH'],  'K_HH'),
        labeled_input('K_NN',  param_dict_1['K_NN'],  'K_NN'),
        labeled_input('I_nut', param_dict_1['I_nut'], 'I_nut'),
        labeled_input('k',     param_dict_1['k'],     'k'),
    ]

    col_layout_2 = [
        labeled_input('DELTA_T',   param_dict_2['DELTA_T'],   'DELTA_T'),
        labeled_input('CCT',       param_dict_2['CCT'],       'CCT'),
        labeled_input('P_P',       param_dict_2['P_P'],       'P_P'),
        labeled_input('MU',        param_dict_2['MU'],        'MU'),
        labeled_input('INIT_CELL', param_dict_2['INIT_CELL'], 'INIT_CELL'),
        labeled_input('P_A',       param_dict_2['P_A'],       'P_A'),
        labeled_input('P_S',       param_dict_2['P_S'],       'P_S'),
        labeled_input('P_MAX',     param_dict_2['P_MAX'],     'P_MAX'),
    ]

    col_layout_3 = [
        labeled_input('total_steps',             param_dict_3['total_steps'],             'total_steps'),
        labeled_input('initial_population',      param_dict_3['initial_population'],      'initial_population'),
        labeled_input('inject_population',       param_dict_3['inject_population'],       'inject_population'),
        labeled_input('division_rate_high_sial', param_dict_3['division_rate_high_sial'], 'division_rate_high_sial'),
        labeled_input('division_rate_low_sial',  param_dict_3['division_rate_low_sial'],  'division_rate_low_sial'),
        labeled_input('death_rate',              param_dict_3['death_rate'],              'death_rate'),
        labeled_input('fusion_rate',             param_dict_3['fusion_rate'],             'fusion_rate'),
        labeled_input('suppressive_effect',      param_dict_3['suppressive_effect'],      'suppressive_effect'),
        labeled_input('permissive_effect',       param_dict_3['permissive_effect'],       'permissive_effect'),
        labeled_input('neutral_effect',          param_dict_3['neutral_effect'],          'neutral_effect'),
        labeled_input('mutation_rate',           param_dict_3['mutation_rate'],           'mutation_rate'),
    ]

    # We place these columns side-by-side, but only one is visible at a time
    column1 = sg.Column(col_layout_1, key='-COL1-', visible=True,  expand_x=True)
    column2 = sg.Column(col_layout_2, key='-COL2-', visible=False, expand_x=True)
    column3 = sg.Column(col_layout_3, key='-COL3-', visible=False, expand_x=True)

    # Main layout
    layout = [
        [sg.Text('Select Simulation Type:')],
        [sg.Combo(
            [
                'Probabilistic Cellular Automata',
                'Model accounting for different types of cancer cells',
                'Model Accounting for Cancer Cell Dive'
            ],
            default_value='Probabilistic Cellular Automata',
            key='-SIM_TYPE-',
            readonly=True,
            enable_events=True,
            size=(40, 1)
        )],
        # Parameter columns all in the same row (only one visible at a time)
        [column1, column2, column3],
        [sg.Button('Submit'), sg.Button('Cancel')]
    ]

    window = sg.Window(
        'Simulation Parameter Setup',
        layout,
        resizable=True  # resizable so that push can reflow properly if needed
    )

    while True:
        event, values = window.read()
        if event in (sg.WIN_CLOSED, 'Cancel'):
            window.close()
            return None

        if event == '-SIM_TYPE-':
            # Show/hide columns based on selected simulation type
            sim_type = values['-SIM_TYPE-']
            if sim_type == 'Probabilistic Cellular Automata':
                window['-COL1-'].update(visible=True)
                window['-COL2-'].update(visible=False)
                window['-COL3-'].update(visible=False)
            elif sim_type == 'Model accounting for different types of cancer cells':
                window['-COL1-'].update(visible=False)
                window['-COL2-'].update(visible=True)
                window['-COL3-'].update(visible=False)
            else:  # 'Model Accounting for Cancer Cell Dive'
                window['-COL1-'].update(visible=False)
                window['-COL2-'].update(visible=False)
                window['-COL3-'].update(visible=True)

        if event == 'Submit':
            # Gather parameters from whichever column is visible
            sim_type = values['-SIM_TYPE-']
            param_dict = {}

            if sim_type == 'Probabilistic Cellular Automata':
                param_dict['K_CC'] = float(values['K_CC'])
                param_dict['K_HH'] = float(values['K_HH'])
                param_dict['K_NN'] = float(values['K_NN'])
                param_dict['I_nut'] = float(values['I_nut'])
                param_dict['k'] = int(values['k'])

            elif sim_type == 'Model accounting for different types of cancer cells':
                param_dict['DELTA_T'] = int(values['DELTA_T'])
                param_dict['CCT'] = int(values['CCT'])
                param_dict['P_P'] = float(values['P_P'])
                param_dict['MU'] = float(values['MU'])
                param_dict['INIT_CELL'] = values['INIT_CELL']
                param_dict['P_A'] = float(values['P_A'])
                param_dict['P_S'] = float(values['P_S'])
                param_dict['P_MAX'] = float(values['P_MAX'])

            else:  
                param_dict['total_steps'] = int(values['total_steps'])
                param_dict['initial_population'] = int(values['initial_population'])
                param_dict['inject_population'] = int(values['inject_population'])
                param_dict['division_rate_high_sial'] = float(values['division_rate_high_sial'])
                param_dict['division_rate_low_sial'] = float(values['division_rate_low_sial'])
                param_dict['death_rate'] = float(values['death_rate'])
                param_dict['fusion_rate'] = float(values['fusion_rate'])
                param_dict['suppressive_effect'] = float(values['suppressive_effect'])
                param_dict['permissive_effect'] = float(values['permissive_effect'])
                param_dict['neutral_effect'] = float(values['neutral_effect'])
                param_dict['mutation_rate'] = float(values['mutation_rate'])

            param_dict['simulation_type'] = sim_type
            window.close()
            return param_dict

if __name__ == "__main__":
    result = get_simulation_params()
    print("Result:", result)
