"""10-step resource benchmark: original aggregate ZCS + central FD second derivatives.

The network, data, first-derivative recovery, optimizer, and profiling protocol
are identical to the pure-AD baseline.  Only the PDE second-derivative stage is
changed from repeated AD to the central finite differences used by PCMON.
"""

from Teacher_operator_phi_2_T_original_ZCS_AD import main


if __name__ == '__main__':
    main(default_second_derivative='fd')
