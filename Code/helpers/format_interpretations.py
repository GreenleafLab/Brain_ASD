def extract_region(region,interp_dict,offset=173,output_length=1000):
    label_prof=interp_dict['label_prof'][region]
    label_sum=round(interp_dict['label_sum'][region],3)
    pred_prof=interp_dict['pred_prof'][region]
    pred_sum=round(interp_dict['pred_sum'][region],3)
    profile_shap=interp_dict['profile_shap'][region][offset:offset+output_length]
    count_shap=interp_dict['count_shap'][region][offset:offset+output_length]
    seq=interp_dict['seq'][region][offset:offset+output_length]
    profile_shap=profile_shap*seq 
    count_shap=count_shap*seq 
    #get the min & max values, to use in ylims for plotting 
    minval_perf=min([label_prof.min(),pred_prof.min()])
    maxval_perf=max([label_prof.max(),pred_prof.max()])
    minval_shap=min([profile_shap.min(), count_shap.min()])
    maxval_shap=max([profile_shap.max(), count_shap.max()])
    return label_prof, label_sum, pred_prof, pred_sum, profile_shap, count_shap, seq, minval_perf, maxval_perf, minval_shap, maxval_shap
