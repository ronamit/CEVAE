# ------- Learn Model -------------#

from __future__ import absolute_import, division, print_function


from learn_standard_CEVAE import learn_standard
from learn_separated_CEVAE import learn_separated
from learn_supervised import learn_supervised
from Utils import matching_estimate, evalaute_effect_estimate
# ----------------------------------------------------------------------------------------#

def learn_latent_model(args, train_set, test_set, anlysis_flag):

    if args.estimation_type == 'proxy_matching':
        est_y0, est_y1 = matching_estimate(train_set['X'], train_set['T'],  train_set['Y'], test_set['X'])
        return evalaute_effect_estimate(est_y0, est_y1, test_set, args, model_name='',
                                 estimation_type=args.estimation_type)

    elif args.model_type == 'standard':
        return learn_standard(args, train_set, test_set, anlysis_flag)

    elif args.model_type in ['separated', 'separated_with_confounder'] :
        return learn_separated(args, train_set, test_set, anlysis_flag)

    elif args.model_type == 'supervised':
        return learn_supervised(args, train_set, test_set)

    else:
        raise ValueError('Unrecognized model_type')
        
# ----------------------------------------------------------------------------------------#
