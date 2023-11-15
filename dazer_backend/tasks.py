import os
import json
from api import models
from api.utils import ATTEMPTS, get_session_subsample_test_folder, get_session_subsample_train_folder, read_file, generate_id, get_model_folder
import dazer
import pandas as pd
from celery import shared_task
from datetime import datetime
import traceback
import sys


@shared_task()
def classification_task(task_id):
    
    task_obj = models.Task.objects.get(token=task_id)
    task_obj.started = datetime.now()
    task_obj.save()
    
    try:
        parameters = json.loads(task_obj.parameters)
        
        subsample_id, classification_task_id = task_id.split('_')
        subsample_obj = models.Subsampling.objects.get(subsample_id=subsample_id)
        target_column = parameters['target_column']
        target_value = str(parameters['target_value']).lower()
        cv = parameters['cv']
        random_states = parameters['random_states']
        filename = subsample_obj.file.filename
        session = subsample_obj.session.session_id
        
        test_folder = get_session_subsample_test_folder(
            session,
            filename,
            subsample_id)
        
        train_folder = get_session_subsample_train_folder(
            session,
            filename,
            subsample_id)
        
        evaluation_list = []
        model_list = [] # used for futher evaluation
        model_keys = [] # used for futher evaluation
        ratios = json.loads(subsample_obj.ratios)
        # every data file is different ratio or random seed
        train_files = os.listdir(train_folder)
        test_files = os.listdir(test_folder)
        subsample_iteration_random_states = json.loads(subsample_obj.iteration_random_states)
        
        progress_max = len(ratios) * len(random_states) * len(subsample_iteration_random_states)
        progress = 0
        
        for subsample_iteration_random_state in subsample_iteration_random_states:
            test_file = [x for x in test_files if f'iteration_random_state={subsample_iteration_random_state}' in x]
            assert len(test_file) == 1
            test_file = test_file[0]
            
            df_test = read_file(os.path.join(test_folder, test_file))
            y_test = (df_test[target_column].map(str).str.lower() == target_value).map(int)
            
            X_test = df_test.drop([target_column], axis=1)
        
            for ratio in ratios:
                data_file = [x for x in train_files if f'ratio={str(ratio)}' in x and f'iteration_random_state={subsample_iteration_random_state}' in x]
                assert len(data_file) == 1
                data_file = data_file[0]
                
                file_path = os.path.join(train_folder, data_file)
                df_train = read_file(file_path)
                y_train = (df_train[target_column].map(str).str.lower() == target_value).map(int)

                X_train = df_train.drop([target_column], axis=1)
                classifier = dazer.Classifier(X_train, y_train, X_test, y_test)
                model_folder_path = get_model_folder(session, filename, classification_task_id)
                
                for random_state in random_states:
                    model_path = os.path.join(model_folder_path, f'ratio={ratio};random_state={random_state};random_state_subsample_iteration={subsample_iteration_random_state}.joblib')
                    model, evaluation = classifier.train_test_random_forest(random_state=random_state, model_path=model_path, scoring='f1', cv=cv, n_jobs=4)
                    evaluation.update(classifier.classifier_prediction_evaluation(models=[model])[0])
                    evaluation.update({'random_state': random_state, 'random_state_subsample_iteration': int(subsample_iteration_random_state), 'ratio': float(ratio)})
                    evaluation_list.append(evaluation)
                    model_list.append(model)
                    model_keys.append(ratio)
                    
                    # update progress
                    progress += 1
                    task_obj.progress = progress / progress_max
                    task_obj.save()
        
        feature_importances = dazer.random_forest_utils.random_forests_feature_importances(models=model_list)
        df_feature_importances = pd.DataFrame(feature_importances)
        feature_importances = df_feature_importances.groupby(model_keys).mean().values
        feature_importances = [list(x) for x in feature_importances]
        
        classification = models.ClassificationTask.objects.create(
            classification_task_id=classification_task_id,
            subsample=subsample_obj,
            cv=cv,
            random_states=json.dumps(random_states),
            evaluation=json.dumps(evaluation_list),
            feature_importances=json.dumps(feature_importances),
            feature_columns=json.dumps(list(X_test.columns)),
            target_column=target_column,
            target_value=target_value)

        task_obj.done = True
        
    except Exception as e:
        print(e)
        # something in task failed
        task_obj.failed = True
        exc_info = sys.exc_info()
        task_obj.status = json.dumps(''.join(traceback.format_exception(*exc_info)))
        
    task_obj.finished = datetime.now()
    task_obj.save()
            
    return