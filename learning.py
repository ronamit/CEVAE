# ------- Learn Model -------------#

from __future__ import absolute_import, division, print_function


from learn_CEVAE import learn_standard
from learn_separated_CEVAE import learn_separated
from learn_supervised import learn_supervised
from Utils import matching_estimate, evalaute_effect_estimate
# ----------------------------------------------------------------------------------------#
def learn_latent_model(args, train_set, test_set):

    if args.estimation_type == 'proxy_matching':
        est_y0, est_y1 = matching_estimate(train_set['X'], train_set['T'],  train_set['Y'], test_set['X'])
        evalaute_effect_estimate(est_y0, est_y1, test_set, model_name='',
                                 estimation_type=args.estimation_type)

    elif args.model_type == 'standard':
        learn_standard(args, train_set, test_set)

    elif args.model_type == 'separated':
        learn_separated(args, train_set, test_set)

    elif args.model_type == 'supervised':
        learn_supervised(args, train_set, test_set)

    else:
        raise ValueError('Unrecognized model_type')
        
# ----------------------------------------------------------------------------------------#
