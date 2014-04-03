from state import *

class ConvFC:
  conv = 0
  conv_fc = 1
  fc = 2

conv_conv_comm_cost = {
    (state0, sisw): lambda input_size, weight_size, overlapping, n: 0,
    (state0, sidw): lambda input_size, weight_size, overlapping, n: 0,
    (state0, disw_b): lambda input_size, weight_size, overlapping, n: 2.0 * (n-1) * weight_size,
    (state0, disw_i): lambda input_size, weight_size, overlapping, n: 2.0 * (n-1) * weight_size,
    
    (sisw, sisw): lambda input_size, weight_size, overlapping, n: 0,
    (sisw, sidw): lambda input_size, weight_size, overlapping, n: 2.0 * input_size * (n-1),
    (sisw, disw_b): lambda input_size, weight_size, overlapping, n: 2.0 * input_size/n * (n-1) + 2.0 * weight_size * (n-1),
    (sisw, disw_i): lambda input_size, weight_size, overlapping, n: 2.0 * input_size/n * (n-1) + 2.0 * weight_size * (n-1) + 2.0 * overlapping,

    (sidw, sisw): lambda input_size, weight_size, overlapping, n: 2.0 * input_size/n * (n-1),
    (sidw, sidw): lambda input_size, weight_size, overlapping, n: 2.0 * input_size/n * (n-1) * 2,
    (sidw, disw_b): lambda input_size, weight_size, overlapping, n: 2.0 * input_size/(n*n) * (n-1) * 2 + 2.0 * weight_size * (n-1),
    (sidw, disw_i): lambda input_size, weight_size, overlapping, n: 2.0 * input_size/(n*n) * (n-1) * 2 + 4.0 * overlapping + 2.0 * weight_size * (n-1),

    (disw_b, sisw): lambda input_size, weight_size, overlapping, n: 2.0 * input_size/n * (n-1),
    (disw_b, sidw): lambda input_size, weight_size, overlapping, n: 2.0 * input_size/n * (n-1) * 2,
    (disw_b, disw_b): lambda input_size, weight_size, overlapping, n: 2.0 * weight_size * (n-1),
    (disw_b, disw_i): lambda input_size, weight_size, overlapping, n: 2.0 * input_size/(n*n) * (n-1) * 2 + 4.0 * overlapping + 2.0 * weight_size * (n-1), 

    (disw_i, sisw): lambda input_size, weight_size, overlapping, n: 2.0 * input_size/n * (n-1),
    (disw_i, sidw): lambda input_size, weight_size, overlapping, n: 2.0 * input_size/n * (n-1) * 2,
    (disw_i, disw_b): lambda input_size, weight_size, overlapping, n: 2.0 * input_size/(n*n) * (n-1) * 2 + 2.0 * weight_size * (n-1),
    (disw_i, disw_i): lambda input_size, weight_size, overlapping, n: 2.0 * overlapping * 2.0 + 2.0 * weight_size * (n-1)
    }

conv_fc_comm_cost = {
    (sisw, sisw): lambda input_size, weight_size, o, n: 0,
    (sisw, sidw_f): lambda input_size, weight_size, o, n: 2.0 * input_size * (n-1),
    (sisw, disw_b): lambda input_size, weight_size, o, n: 2.0 * input_size/n * (n-1) + 2.0 * weight_size * (n-1),
    
    (sidw, sisw): lambda input_size, weight_size, o, n: 2.0 * input_size/n * (n-1),
    (sidw, sidw_f): lambda input_size, weight_size, o, n: 2.0 * input_size/n * (n-1) * 2,
    (sidw, disw_b): lambda input_size, weight_size, o, n: 2.0 * input_size/(n*n) * (n-1) + 2.0 * weight_size * (n-1),

    (disw_b, sisw): lambda input_size, weight_size, o, n: 2.0 * input_size/n * (n-1),
    (disw_b, sidw_f): lambda input_size, weight_size, o, n: 2.0 * input_size/n * (n-1) * 2,
    (disw_b, disw_b): lambda input_size, weight_size, o, n: 2.0 * weight_size * (n-1),

    (disw_i, sisw): lambda input_size, weight_size, o, n: 2.0 * input_size/n * (n-1),
    (disw_i, sidw_f): lambda input_size, weight_size, o, n: 2.0 * input_size/n * (n-1) * 2,
    (disw_i, disw_b): lambda input_size, weight_size, o, n: 2.0 * input_size/(n*n) * (n-1) * 2 + 2.0 * weight_size * (n-1)
    }

fc_fc_comm_cost = {
    (sisw, sisw): lambda input_size, weight_size, o, n: 0,
    (sisw, sidw_f): lambda input_size, weight_size, o, n: 2.0 * input_size * (n-1),
    (sisw, disw_b): lambda input_size, weight_size, o, n: 2.0 * input_size/n * (n-1) + 2.0 * weight_size * (n-1),

    (sidw_f, sisw): lambda input_size, weight_size, o, n: 2.0 * input_size/n * (n-1),
    (sidw_f, sidw_f): lambda input_size, weight_size, o, n: 2.0 * input_size/n * (n-1) * 2,
    (sidw_f, disw_b): lambda input_size, weight_size, o, n: 2.0 * input_size/(n*n) * (n-1) * 2 + 2.0 * weight_size * (n-1),

    (disw_b, sisw): lambda input_size, weight_size, o, n: 2.0 * input_size/n * (n-1),
    (disw_b, sidw_f): lambda input_size, weight_size, o, n: 2.0 * input_size/n * (n-1) * 2,
    (disw_b, disw_b): lambda input_size, weight_size, o, n : 2.0 * weight_size * (n-1)
    }

comm_cost = {ConvFC.conv:conv_conv_comm_cost, ConvFC.conv_fc:conv_fc_comm_cost,
    ConvFC.fc:fc_fc_comm_cost}
