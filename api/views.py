from django.shortcuts import render
from rest_framework.response import Response
from rest_framework.views import APIView, View
from rest_framework.parsers import MultiPartParser, FileUploadParser, JSONParser
from rest_framework.decorators import parser_classes, api_view
from pathlib import Path
import os
import random
import string
import dazer
import pandas as pd
from django.http import HttpResponseBadRequest
from api import models, serializers
import json
import numpy.random as npr
import shutil
from collections.abc import MutableMapping
from collections import defaultdict


ATTEMPTS = 300


def generate_id(size):
    chars = string.ascii_uppercase + string.digits
    return ''.join(random.choice(chars) for _ in range(size))

def get_session_files_folder(session):
    return os.path.join("_sessions", session, 'files')

def get_session_subsample_folder(session):
    return os.path.join("_sessions", session, 'subsamples')

def get_session_subsample_task_folder(session, filename, subsample_id):
    return os.path.join("_sessions", session, 'subsamples', filename, subsample_id)

def get_session_subsample_test_folder(session, filename, subsample_id):
    return os.path.join("_sessions", session, 'subsamples', filename, subsample_id, 'test')

def get_session_subsample_train_folder(session, filename, subsample_id):
    return os.path.join("_sessions", session, 'subsamples', filename, subsample_id, 'train')

def get_model_folder(session, filename, classification_task_id):
    return os.path.join("_sessions", session, 'models', filename, classification_task_id, 'train')

def read_file(filename):
    df = pd.read_csv(filename, index_col=0, sep='\t')
    return df

def write_file(df, filename):
    df.to_csv(filename, sep='\t')
    return

def clean_input_dataframe(df):
    rows_removed = 0
    length = len(df.index)
    # drop rows containing NA
    df = df.dropna()
    if length > len(df.index):
        rows_removed = length - len(df.index)
    return df, rows_removed

def get_param_from_filename(target, filename, sep=';'):
    filename = filename.split(os.sep)[-1]
    # remove file ending
    filename = filename[:filename.rfind('.')]
    for param in filename.split(sep):
        if '=' not in param:
            continue
        key, value = param.split('=')
        if key == target:
            return value
    return ''

def get_df_column_information(df):
    columns = list(df.columns)
    categorical_columns = df.columns[~df.columns.isin(df._get_numeric_data().columns)]
    categorical_columns_values = {}
    for col in categorical_columns:
        categorical_columns_values[col] = df[col].dropna().unique().tolist()
    return columns, categorical_columns_values


@parser_classes([FileUploadParser])
class FileUpload(APIView):

    def put(self, request, session, filename):
        print(filename)
        content = request.body
        if content is None:
            return Response({})
        
        session_obj, _ = models.Session.objects.get_or_create(session_id=session)
        
        if len(models.File.objects.filter(session=session_obj, filename=filename)):
            # file already exists, increment filename
            for i in range(1, ATTEMPTS+1):
                filename_incremented = f'{filename[:-4]}{i}.tsv'
                if not len(models.File.objects.filter(session=session_obj, filename=filename_incremented)):
                    break
            if i == ATTEMPTS:
                return HttpResponseBadRequest('Filename already taken.')
            filename = filename_incremented
        
        Path(get_session_files_folder(session)).mkdir(parents=True, exist_ok=True)
        file_path = os.path.join(get_session_files_folder(session), filename)
        
        print(file_path)
        
        with open(file_path, "wb+") as destination:
            destination.write(content)
            
        try:
            df = read_file(file_path)
            df, rows_removed = clean_input_dataframe(df)
            columns, categorical_columns_values = get_df_column_information(df)
            print(columns, categorical_columns_values )
            write_file(df, file_path)
        except:
            return HttpResponseBadRequest('Could not read file.')
        
        # create session instance on file upload
        try:
            models.File.objects.create(session=session_obj, filename=filename, rows_removed=rows_removed, columns=json.dumps(columns), categorical_columns_values=json.dumps(categorical_columns_values))
        except Exception as e:
            print(e)
            print('Could not create file object')
            # file already exists for this session
            return HttpResponseBadRequest('File already exists.')
        
        return Response({})
    

class File(APIView):
    def get(self, request, subsample_id):
        subsample_obj = models.Subsampling.objects.get(subsample_id=subsample_id)
        file = serializers.FileSerializer().to_representation(subsample_obj.file)
        return Response({'file': file})
    
    
class FileColumns(APIView):
    def get(self, request, session, filename):
        file_obj = models.File.objects.get(session__session_id=session, filename=filename)
        return Response({'columns': json.loads(file_obj.columns), 'categorical_columns_values': json.loads(file_obj.categorical_columns_values)})
    

class FileColumnsSubsample(APIView):
    def get(self, request, subsample_id):
        subsample_obj = models.Subsampling.objects.get(subsample_id=subsample_id)
        file_obj = subsample_obj.file
        return Response({'columns': json.loads(file_obj.columns), 'categorical_columns_values': json.loads(file_obj.categorical_columns_values)})


class SessionId(APIView):
    # Session instance is created only on file upload
    
    def get(self, request):
        session = None
        while True:
            session = generate_id(10)
            if not os.path.isdir(os.path.join("_sessions", session)):
                break

        if session is None:
            return Response({'session': ''})

        return Response({'session': session})
    
    
class SessionFiles(APIView):
    
    def get(self, request, session):
        session_files_folder = get_session_files_folder(session)
        if not os.path.isdir(session_files_folder):
            return Response({'sessionFiles': []})
        file_obj_list = models.File.objects.filter(session__session_id=session)
        session_files = serializers.FileSerializer(many=True).to_representation(file_obj_list)
        
        session_file_data = []
        for session_file in session_files:
            subsample_task_obj_list = models.Subsampling.objects.filter(session__session_id=session, file__filename=session_file['filename'])
            
            subsample_task_information = []
            for subsample_task_obj in subsample_task_obj_list:
                classification_task_obj_list = models.ClassificationTask.objects.filter(subsample=subsample_task_obj)
                
                classification_task_information = []
                for classification_task_obj in classification_task_obj_list:
                    classification_task_information.append({
                        'classificationId': classification_task_obj.classification_task_id
                    })
                subsample_task_information.append({
                    'subsampleTaskId': subsample_task_obj.subsample_id,
                    'classificationTasks': classification_task_information
                    })

            session_file_data.append({
                **session_file, 
                'subsampleTasks': subsample_task_information
            })
                
        return Response({'sessionFiles': session_file_data})


def flatten_dict(dictionary, parent_key='', separator='_'):
    items = []
    for key, value in dictionary.items():
        new_key = parent_key + separator + key if parent_key else key
        if isinstance(value, MutableMapping):
            items.extend(flatten_dict(value, new_key, separator=separator).items())
        else:
            items.append((new_key, value))
    return dict(items)
        
def merge_flat_dicts(array_of_dicts):
    keys = array_of_dicts[0].keys()
    return {key: sum([e[key] for e in array_of_dicts])/len(array_of_dicts) for key in keys}


@parser_classes([JSONParser])
class Subsample(APIView):
    
    def post(self, request, session, filename):
        random_states = [101, 102, 103, 104, 105]
        
        session_obj = models.Session.objects.get(session_id=session)
        file_obj = models.File.objects.get(session=session_obj, filename=filename)
        
        keep_ratio_columns = request.data.get('keepRatioColumns')
        test_ratio = request.data.get('testRatio')
        ratios = request.data.get('ratios', [.5, 1])
        n_random_states = int(request.data.get('nRandomStates', 1))

        allowed_deviation = float(request.data.get('allowed_deviation', 0.2))

        iteration_random_states = random_states[:n_random_states]
        
        #### move this to standalone task
        for _ in range(ATTEMPTS):
            subsample_id = generate_id(10)
            if not len(models.Subsampling.objects.filter(subsample_id=subsample_id)):
                break
            
        folder_test = get_session_subsample_test_folder(session, filename, subsample_id)
        Path(folder_test).mkdir(parents=True, exist_ok=True)
        
        df = read_file(os.path.join(get_session_files_folder(session), filename))
        subsampler = dazer.Subsampler(df, keep_ratio_columns, allowed_deviation=allowed_deviation)
        for random_state in range(1, ATTEMPTS+1):
            df_test = subsampler.extract_test(test_size=0.2, random_state=random_state)
            if df_test is not None:
                break
        df_test.to_csv(os.path.join(folder_test, f'type=test;ratio={str(test_ratio)};random_state={random_state}.tsv'), sep='\t')
        
        folder_train = get_session_subsample_train_folder(session, filename, subsample_id)
        Path(folder_train).mkdir(parents=True, exist_ok=True)
        
        for seed in iteration_random_states:
            npr.seed(seed)
            for attempt in range(1, ATTEMPTS+1):
                random_state = npr.randint(1, 999999999)
                for ratio in ratios:
                    df_train = subsampler.subsample(ratio, random_state)
                    if df_train is None:
                        # jump to next random state
                        break
                    df_train.to_csv(os.path.join(folder_train, f'type=train;ratio={str(ratio)};iteration_random_state={seed};random_state={random_state}.tsv'), sep='\t')
                if ratio == 1:
                    break
                
        ### SUBSAMPLING DONE ABOVE, FORMAT OUTPUT BELOW
        
        # train data
        folder_train = get_session_subsample_train_folder(session, filename, subsample_id)
        data_all_random_iterations = []
        for iteration_state in iteration_random_states:
            ratios_data = {}
            for file in os.listdir(folder_train):
                if f'iteration_random_state={iteration_state}' not in file:
                    continue
                for param in file.split(';'):
                    key, value = param.split('=')
                    if key == 'ratio':
                        ratios_data[value] = file
            data = {}
            for ratio, path in ratios_data.items():
                df = read_file(os.path.join(folder_train, path))
                data[f'train:{ratio}'] = {}
                for col in keep_ratio_columns:
                    data[f'train:{ratio}'][col] = df[col].value_counts().to_dict()
                        
            # test data
            folder_test = get_session_subsample_test_folder(session, filename, subsample_id)
            file = os.listdir(folder_test)[0]
            test_ratio = ''
            for param in file.split(';'):
                key, value = param.split('=')
                if key == 'ratio':
                    test_ratio = value
            df = read_file(os.path.join(folder_test, file))
            data[f'test:{test_ratio}'] = {}
            for col in keep_ratio_columns:
                data[f'test:{test_ratio}'][col] = df[col].value_counts().to_dict()
            
            # reformat data for plotting
            data_reformat = {col: {} for col in keep_ratio_columns}
            for ratio_category, ratio_data in data.items():
                for keep_ratio_col, value_counts in ratio_data.items():
                    if 'test' in ratio_category:
                        data_reformat[keep_ratio_col][ratio_category] = value_counts
                    else:
                        ratio = ratio_category.split(':')[1]
                        data_reformat[keep_ratio_col][ratio] = value_counts
            data_all_random_iterations.append(data_reformat)
            
        mean_dict = defaultdict(lambda: defaultdict(dict))
        for key, mean in merge_flat_dicts([flatten_dict(d) for d in data_all_random_iterations]).items():
            feature, ratio, value = key.split('_')
            mean_dict[feature][ratio][value] = mean
        
        # get categorical columns
        categorical_columns = df.columns[~df.columns.isin(df._get_numeric_data().columns)]
        unique_values = {}
        for col in categorical_columns:
            unique_values[col] = df[col].dropna().unique().tolist()
            
        ### RESULT FORMATTING DONE
        
        ### END move this to standalone task
        
        # test_file = os.listdir(get_session_subsample_test_folder(session, filename, subsample_id))
        # train_files = os.listdir(get_session_subsample_train_folder(session, filename, subsample_id))
            
        models.Subsampling.objects.create(
            session=session_obj,
            file=file_obj,
            subsample_id=subsample_id,
            keep_ratio_columns=json.dumps(keep_ratio_columns),
            ratios=json.dumps(ratios_data),
            iteration_random_states=json.dumps(random_states),
            test_ratio=test_ratio,
            allowed_deviation=allowed_deviation,
            result_formatted =json.dumps({'data': dict(mean_dict), 'filename': filename, 'keepRatioColumns': keep_ratio_columns, 'ratios': ratios, 'testLabel': f'test:{test_ratio}'})
            )
            
        return Response()
        

class SubsampleResult(APIView):
    
    def get(self, request, subsample_id):
        subsample_obj = models.Subsampling.objects.get(subsample_id=subsample_id)
        return Response({**json.loads(subsample_obj.result_formatted), 'allowedDeviation': subsample_obj.allowed_deviation, 'iterationRandomStates': json.loads(subsample_obj.iteration_random_states), 'testRatio': subsample_obj.test_ratio})


@parser_classes([JSONParser])
class Classification(APIView):
    
    def post(self, request, subsample_id):
        
        ###### OUTSOURCE
        random_states = [101, 102, 103, 104, 105]
        subsample_obj = models.Subsampling.objects.get(subsample_id=subsample_id)
        target_column = request.data.get('targetColumn')
        target_value = request.data.get('targetValue')
        cv = int(request.data.get('crossValidationK', 2))
        n_random_states = int(request.data.get('nRandomStates', 1))

        if not (1 <= n_random_states <= 5):
            return HttpResponseBadRequest('nRandomStates must be a value between 1 and 5')
        random_states = random_states[:n_random_states]

        if not (1 <= cv <= 10):
            return HttpResponseBadRequest('crossValidationK must be a value between 1 and 10')
        
        filename = subsample_obj.file.filename
        session = subsample_obj.session.session_id
        
        for _ in range(ATTEMPTS):
            classification_task_id = generate_id(10)
            if not len(models.ClassificationTask.objects.filter(classification_task_id=classification_task_id)):
                break
        
        test_folder = get_session_subsample_test_folder(
            session,
            filename,
            subsample_id)
        
        train_folder = get_session_subsample_train_folder(
            session,
            filename,
            subsample_id)
        
        df_test = read_file(os.path.join(test_folder, os.listdir(test_folder)[0]))
        y_test = (df_test[target_column].str.lower() == target_value).map(int)
        X_test = df_test.drop([target_column], axis=1)
        
        evaluation_list = []
        model_list = [] # used for futher evaluation
        model_keys = [] # used for futher evaluation
        ratios = []
        # every data file is different ratio or random seed
        for data_file in os.listdir(train_folder):
            file_path = os.path.join(train_folder, data_file)
            df_train = read_file(file_path)
            
            y_train = (df_train[target_column].str.lower() == target_value).map(int)
            X_train = df_train.drop([target_column], axis=1)
            classifier = dazer.Classifier(X_train, y_train, X_test, y_test)
            model_folder_path = get_model_folder(session, filename, classification_task_id)
            
            ratio = get_param_from_filename('ratio', data_file)
            random_state_data = get_param_from_filename('random_state', data_file)
            
            for random_state in random_states:
                model_path = os.path.join(model_folder_path, f'ratio={ratio};random_state={random_state};random_state_data={random_state_data}.joblib')
                model, evaluation = classifier.train_test_random_forest(random_state=random_state, model_path=model_path, scoring='f1', cv=cv)
                evaluation.update(classifier.classifier_prediction_evaluation(models=[model])[0])
                evaluation.update({'random_state': random_state, 'random_state_data': int(random_state_data), 'ratio': float(ratio)})
                evaluation_list.append(evaluation)
                model_list.append(model)
                model_keys.append(ratio)

            ratios.append(ratio)
        
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
             
        # DONE
        return Response({'classification_task_id': classification_task_id})
    
    
def _merge_classification_results(json_string):
    # merge results of different random to calculate average in backend since it is more convenient here.
    # Is there a nice way to visualize the single points of the random states in the frontend?
    array = json.loads(json_string)
    df = pd.DataFrame(array)
    return df.groupby('ratio').mean().reset_index().to_dict(orient='records')
    
class ClassificationResult(APIView):    
    
    def get(self, request, classification_task_id):
        classification_task = models.ClassificationTask.objects.get(classification_task_id=classification_task_id)
        evaluation_list = _merge_classification_results(classification_task.evaluation)
        feature_importances = json.loads(classification_task.feature_importances)
        feature_importances_features = json.loads(classification_task.feature_columns)
        
        if len(feature_importances_features) > 50:
            # get most important features
            df_feature_importances = pd.DataFrame(feature_importances, columns=feature_importances_features)
            feature_importances_features = df_feature_importances.mean().sort_values(ascending=False).index[:50]
            feature_importances = [list(x) for x in df_feature_importances[feature_importances_features].values]

        return Response({
            'data': evaluation_list,
            'feature_importances': feature_importances,
            'filename': classification_task.subsample.file.filename,
            'target_column': classification_task.target_column,
            'target_value': classification_task.target_value,
            'feature_importances_features': feature_importances_features,
            'crossValidationK': classification_task.cv,
            'randomStates': json.loads(classification_task.random_states),
            })


def delete_classification_task(classification_task_id):
    classification_task = models.ClassificationTask.objects.get(classification_task_id=classification_task_id)
    model_folder = get_model_folder(classification_task.subsample.session.session_id, classification_task.subsample.file.filename, classification_task_id)
    # delete folder with models
    shutil.rmtree(model_folder)
    # delete db entry
    classification_task.delete()
    
    
class ClassificationDelete(APIView):    
    
    def get(self, request, classification_task_id):
        delete_classification_task(classification_task_id)
        return Response()
        

def delete_subsample_task(subsample_id):
    subsample_obj = models.Subsampling.objects.get(subsample_id=subsample_id)
    
    for classifcation_task in subsample_obj.classification:
        delete_classification_task(classifcation_task.classification_task_id)
    
    subsample_folder = get_session_subsample_task_folder(subsample_obj.session.session_id, subsample_obj.file.filename, subsample_id)
    # delete folder with subsample results
    shutil.rmtree(subsample_folder)
    # delete db entry
    subsample_obj.delete()


class SubsampleDelete(APIView):    
    
   def get(self, request, subsample_id):
        delete_subsample_task(subsample_id)
        return Response()
    
    
def delete_file(session, filename):
    file_obj = models.File.objects.get(session__session_id=session, filename=filename)
    
    for subsample in file_obj.subsamples:
        delete_subsample_task(subsample.subsample_id)
    
    session_files_folder = get_session_files_folder(session)
    # delete delete uploaded file
    os.remove(os.path.join(session_files_folder, filename))
    # delete db entry
    file_obj.delete()
    
    
class FileDelete(APIView):    
    
    def get(self, request, session, filename):
        delete_file(session, filename)
        return Response()
    

def delete_session(session):
    session_obj = models.Session.objects.get(session_id=session)
    
    for file_obj in session_obj.files:
        delete_file(file_obj.filename)
    
    session_files_folder = get_session_files_folder(session)
    # delete delete all uploaded files
    shutil.rmtree(session_files_folder)
    # delete db entry
    session_obj.delete()


class SessionDelete(APIView):    
    
    def get(self, request, session):
        delete_session(session)
        return Response()