# Input parameters for gene sets, read docs for information on what they do
descDict1 = {'lower': 1,
             'upper': 5,
             'centres': '{8, 7, 6, 1, 16, 17, 9}',
             'neighbours': '{8, 7, 6, 1, 16, 17, 9}',
             'mu': 0,
             'mu_hat': 0,
             'nu': 2,
             'nu_hat': 0,
             'mutation_chance': 0.50,
             'min_cutoff': 1,
             'max_cutoff': 5,
             'min_sigma': 0.1,
             'max_sigma': 0.9,
             'message_steps': 0}


population_parameters = {'best_sample': 4,
                         'lucky_few': 2,
                         'population_size': 12,
                         'number_of_children': 4,
                         'maximise_scores': True}


history_parameters = {'early_stop': 2,
                      'early_number': 3,
                      'min_generations': 5}

# List of parameters, you might want multiple GeneSets if you are doing things
# like double soaps.
descList = [descDict1]
num_gens = 6