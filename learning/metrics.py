import numpy as np


def summarize(predictions,targets,affected):
    err=(predictions-targets)**2
    return {'mse':float(err.mean()),'final_mse':float(err[:,-1].mean()),
            'affected_final_mse':float(err[affected,-1].mean()),'unaffected_final_mse':float(err[~affected,-1].mean()),
            'delta_mse':float(((predictions[:,1]-predictions[:,0])-(targets[:,1]-targets[:,0])) .__pow__(2).mean()),
            'accuracy_0.1':float((abs(predictions[:,-1]-targets[:,-1])<=.1).mean())}


def revision_gate(baseline,candidate,targets,affected,tolerance=.01):
    b=summarize(baseline,targets,affected);c=summarize(candidate,targets,affected)
    da=c['affected_final_mse']-b['affected_final_mse'];du=c['unaffected_final_mse']-b['unaffected_final_mse']
    return {'pass':bool(da<0 and du<=tolerance),'affected_change':da,'unaffected_final_change':du,'tolerance':tolerance}
