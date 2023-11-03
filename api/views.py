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


ATTEMPTS = 300


def generate_id(size):
    chars = string.ascii_uppercase + string.digits
    return ''.join(random.choice(chars) for _ in range(size))

def get_session_files_folder(session):
    return os.path.join("_sessions", session, 'files')

def get_session_subsample_folder(session):
    return os.path.join("_sessions", session, 'subsamples')

def get_session_subsample_task_folder(session, subsample_id):
    return os.path.join("_sessions", session, 'subsamples', subsample_id)

def get_session_subsample_test_folder(session, filename, subsample_id):
    return os.path.join("_sessions", session, 'subsamples', filename, subsample_id, 'test')

def get_session_subsample_train_folder(session, filename, subsample_id):
    return os.path.join("_sessions", session, 'subsamples', filename, subsample_id, 'train')

def get_model_folder(session, filename, subsample_id):
    return os.path.join("_sessions", session, 'models', filename, subsample_id, 'train')

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
            return Response({'sessionFiles': [], 'subsampleData': {}})
        file_obj_list = models.File.objects.filter(session__session_id=session)
        session_files = serializers.FileSerializer(many=True).to_representation(file_obj_list)
        
        subsample_data = {}
        for filename in os.listdir(session_files_folder):
            session_subsamples_folder = get_session_subsample_folder(session)
            if os.path.isdir(os.path.join(session_subsamples_folder, filename)):
                subsample_data[filename] = os.listdir(os.path.join(session_subsamples_folder, filename))
                
        return Response({'sessionFiles': session_files, 'subsampleData': subsample_data})


@parser_classes([JSONParser])
class Subsample(APIView):
    
    def post(self, request, session, filename):
        session_obj = models.Session.objects.get(session_id=session)
        file_obj = models.File.objects.get(session=session_obj, filename=filename)
        
        keep_ratio_columns = request.data.get('keepRatioColumns')
        ratio = request.data.get('ratio')
        
        attempts = 100
        ratios = [.2, .6, 1]
        
        #### move this to standalone task
        session_files_folder = get_session_files_folder(session)
        session_subsample_folder = get_session_subsample_folder(session)
        
        for _ in range(attempts):
            subsample_id = generate_id(10)
            if not len(models.Subsampling.objects.filter(subsample_id=subsample_id)):
                break
            # if not os.path.isdir(os.path.join(session_subsample_folder, filename, subsample_id)):
            #     break
            
        folder_test = get_session_subsample_test_folder(session, filename, subsample_id)
        Path(folder_test).mkdir(parents=True, exist_ok=True)
        
        df = read_file(os.path.join(get_session_files_folder(session), filename))
        subsampler = dazer.Subsampler(df, keep_ratio_columns, allowed_deviation=0.9)
        for random_state in range(1, attempts+1):
            df_test = subsampler.extract_test(test_size=0.2, random_state=random_state)
            if df_test is not None:
                break
        df_test.to_csv(os.path.join(folder_test, f'type=test;ratio={str(ratio)};random_state={random_state}.tsv'), sep='\t')
        
        folder_train = get_session_subsample_train_folder(session, filename, subsample_id)
        Path(folder_train).mkdir(parents=True, exist_ok=True)
        
        for random_state in range(1, attempts+1):
            for ratio in ratios:
                df_train = subsampler.subsample(ratio, random_state)
                if df_train is None:
                    # jump to next random state
                    break
                df_train.to_csv(os.path.join(folder_train, f'type=train;ratio={str(ratio)};random_state={random_state}.tsv'), sep='\t')
            if ratio == 1:
                break
        ### END move this to standalone task
            
        # read subsampling result file lists
        test_file = os.listdir(get_session_subsample_test_folder(session, filename, subsample_id))
        train_files = os.listdir(get_session_subsample_train_folder(session, filename, subsample_id))
        
        models.Subsampling.objects.create(
            session=session_obj, 
            file=file_obj, 
            subsample_id=subsample_id,
            keep_ratio_columns=json.dumps(keep_ratio_columns)
            )
            
        return Response({'testFile': test_file, 'trainFiles': train_files})
        

class SubsampleResult(APIView):
    
    def get(self, request, subsample_id):
        # session_obj = models.Session.objects.get(session_id=session_id)
        subsample_obj = models.Subsampling.objects.get(subsample_id=subsample_id)
        session_id = subsample_obj.session.session_id
        filename = subsample_obj.file.filename
        
        keep_ratio_columns = json.loads(subsample_obj.keep_ratio_columns)
        
        # train data
        ratios = {}
        folder_train = get_session_subsample_train_folder(session_id, filename, subsample_id)
        for file in os.listdir(folder_train):
            for param in file.split(';'):
                key, value = param.split('=')
                if key == 'ratio':
                    ratios[value] = file
        data = {}
        for ratio, path in ratios.items():
            df = read_file(os.path.join(folder_train, path))
            data[f'train:{ratio}'] = {}
            for col in keep_ratio_columns:
                data[f'train:{ratio}'][col] = df[col].value_counts().to_dict()
                    
        # test data
        folder_test = get_session_subsample_test_folder(session_id, filename, subsample_id)
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
        
        # get categorical columns
        categorical_columns = df.columns[~df.columns.isin(df._get_numeric_data().columns)]
        unique_values = {}
        for col in categorical_columns:
            unique_values[col] = df[col].dropna().unique().tolist()
                    
        return Response({'data': data_reformat, 'filename': filename, 'keepRatioColumns': keep_ratio_columns, 'ratios': ratios.keys(), 'testLabel': f'test:{test_ratio}', 'testRatio': test_ratio})


@parser_classes([JSONParser])
class Classification(APIView):
    
    def post(self, request, subsample_id):
        
        ###### OUTSOURCE
        random_states = [101]
        subsample_obj = models.Subsampling.objects.get(subsample_id=subsample_id)
        target_column = request.data.get('targetColumn')
        target_value = request.data.get('targetValue')
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
        model_paths = [] # used for futher evaluation
        # every data file is different ratio or random seed
        for data_file in os.listdir(train_folder):
            file_path = os.path.join(train_folder, data_file)
            df_train = read_file(file_path)
            
            y_train = (df_train[target_column].str.lower() == target_value).map(int)
            X_train = df_train.drop([target_column], axis=1)
            classifier = dazer.Classifier(X_train, y_train, X_test, y_test)
            model_folder_path = get_model_folder(session, filename, subsample_id)
            
            ratio = get_param_from_filename('ratio', data_file)
            random_state_data = get_param_from_filename('random_state', data_file)
            
            for random_state in random_states:
                model_path = os.path.join(model_folder_path, f'ratio={ratio};random_state={random_state};random_state_data={random_state_data}.joblib')
                _, evaluation = classifier.train_test_random_forest(random_state=random_state, model_path=model_path, scoring='f1', cv=2)
                evaluation.update({'random_state': random_state, 'random_state_data': random_state_data, 'ratio': ratio})
                evaluation_list.append(evaluation)
                model_paths.append(model_path)
        
        feature_importances = dazer.random_forest_utils.random_forests_feature_importances_from_files(model_paths)
        
        classification = models.ClassificationTask.objects.create(
            classification_task_id=classification_task_id,
            subsample=subsample_obj,
            evaluation=json.dumps(evaluation_list),
            feature_importances=json.dumps(feature_importances),
            feature_columns=json.dumps(list(X_test.columns)),
            target_column=target_column,
            target_value=target_value)
             
        # DONE
        return Response({'classification_task_id': classification_task_id})
    
    
class ClassificationResult(APIView):    
    
    def get(self, request, classification_task_id):
        classification_task = models.ClassificationTask.objects.get(classification_task_id=classification_task_id)
        evaluation_list = json.loads(classification_task.evaluation)
        feature_importances = json.loads(classification_task.feature_importances)
        return Response({
            'data': evaluation_list,
            'feature_importances': feature_importances,
            'filename': classification_task.subsample.file.filename,
            'target_column': classification_task.target_column,
            'target_value': classification_task.target_value,
            'features': json.loads(classification_task.feature_columns),
            })
