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
from api import models

ATTEMPTS = 100


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

def read_file(filename):
    df = pd.read_csv(filename, index_col=0, sep='\t')
    return df


@parser_classes([FileUploadParser])
class FileUpload(APIView):

    def put(self, request, session, filename):
        content = request.body
        if content is None:
            return Response({})
        
        Path(get_session_files_folder(session)).mkdir(parents=True, exist_ok=True)
        
        with open(os.path.join(get_session_files_folder(session), filename), "wb+") as destination:
            destination.write(content)
            
        try:
            df = read_file(os.path.join(get_session_files_folder(session), filename))
        except:
            return HttpResponseBadRequest('Could not read file.')
        
        # create session instance on file upload
        session_obj, created = models.Session.objects.get_or_create(session_id=session)
        try:
            models.File.objects.create(session=session_obj, filename=filename)
        except:
            # file already exists for this session
            return HttpResponseBadRequest('File already exists.')
        
        return Response({})
    

class FileColumns(APIView):
    def get(self, request, session, filename):
        session_files_folder = get_session_files_folder(session)
        df = read_file(os.path.join(session_files_folder, filename))
        columns = df.columns
        
        categorical_columns = df.columns[~df.columns.isin(df._get_numeric_data().columns)]
        unique_values = {}
        for col in categorical_columns:
            unique_values[col] = df[col].dropna().unique().tolist()
        return Response({'columns': columns, 'categoricalColumns': unique_values})


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
            return Response({'sessionFiles': [], 'subsampleData': subsample_data})
        session_files = os.listdir(session_files_folder)
        
        subsample_data = {}
        for filename in session_files:
            session_subsamples_folder = get_session_subsample_folder(session)
            if os.path.isdir(os.path.join(session_subsamples_folder, filename)):
                subsample_data[filename] = os.listdir(os.path.join(session_subsamples_folder, filename))
                
        return Response({'sessionFiles': session_files, 'subsampleData': subsample_data})


@parser_classes([JSONParser])
class Subsample(APIView):
    
    def post(self, request, session, filename):
        session_obj = models.Session.objects.get(session_id=session)
        file_obj = models.File.objects.get(session=session_obj, filename=filename)
        
        target_column = request.data.get('targetCol')
        # target_value = request.POST.get('targetVal')
        keep_ratio_columns = request.data.get('keepRatioColumns')
        
        if target_column not in keep_ratio_columns:
            keep_ratio_columns.append(target_column)

        ratio = request.data.get('ratio')
        
        attempts = 100
        ratios = [.2, .3, .4, .5, .6, .7, .8, .9, 1]
        
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
        subsampler = dazer.Subsampler(df, target_column, keep_ratio_columns, allowed_deviation=0.9)
        for random_state in range(1, attempts+1):
            try:
                df_test = subsampler.extract_test(test_size=0.2, random_state=random_state)
                break
            except Exception as e:
                print(e)
                continue
        df_test.to_csv(os.path.join(folder_test, f'type=test;ratio={str(ratio)};seed={random_state}.tsv'), sep='\t')
        
        folder_train = get_session_subsample_train_folder(session, filename, subsample_id)
        Path(folder_train).mkdir(parents=True, exist_ok=True)
        
        for random_state in range(1, attempts+1):
            for ratio in ratios:
                try:
                    df_train = subsampler.subsample(ratio, random_state)
                except:
                    continue
                df_train.to_csv(os.path.join(folder_train, f'type=train;ratio={str(ratio)};seed={random_state}.tsv'), sep='\t')
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
            target_column=target_column,
            keep_ratio_columns='\t'.join(keep_ratio_columns)
            )
            
        return Response({'test_file': test_file, 'train_files': train_files})
        

class SubsampleResult(APIView):
    
    def get(self, request, subsample_id):
        # session_obj = models.Session.objects.get(session_id=session_id)
        subsample_obj = models.Subsampling.objects.get(subsample_id=subsample_id)
        session_id = subsample_obj.session.session_id
        filename = subsample_obj.file.filename
        
        print('session_id', session_id)
        print('filename', filename)
        
        keep_ratio_columns = subsample_obj.keep_ratio_columns.split('\t')
        
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
            data[f'train_{ratio}'] = {}
            for col in keep_ratio_columns:
                data[f'train_{ratio}'][col] = df[col].value_counts().to_dict()
                    
        # test data
        folder_test = get_session_subsample_test_folder(session_id, filename, subsample_id)
        file = os.listdir(folder_test)[0]
        test_ratio = ''
        for param in file.split(';'):
            key, value = param.split('=')
            if key == 'ratio':
                test_ratio = value
        df = read_file(os.path.join(folder_test, file))
        data[f'test_{test_ratio}'] = {}
        for col in keep_ratio_columns:
            data[f'test_{test_ratio}'][col] = df[col].value_counts().to_dict()
        
        # reformat data for plotting
        data_reformat = {col: {} for col in keep_ratio_columns}
        for ratio_category, ratio_data in data.items():
            for keep_ratio_col, value_counts in ratio_data.items():
                if 'test' in ratio_category:
                    data_reformat[keep_ratio_col][ratio_category] = value_counts
                else:
                    ratio = ratio_category.split('_')[1]
                    data_reformat[keep_ratio_col][ratio] = value_counts
                    
        return Response({'data': data_reformat, 'filename': filename, 'keepRatioColumns': keep_ratio_columns, 'ratios': ratios.keys(), 'testLabel': f'test_{test_ratio}', 'testRatio': test_ratio})



@parser_classes([JSONParser])
class Classification(APIView):
    
    def post(self, request, subsample_id):
        
        subsample_obj = models.Subsampling.objects.get(subsample_id=subsample_id)
        target_column = subsample_obj.target_column
        
        for _ in range(ATTEMPTS):
            classification_id = generate_id(10)
            if not len(models.Classification.objects.filter(classification_id=classification_id)):
                break
        classification = models.Classification.objects.create(classification_id=classification_id, subsample=subsample_obj)
        
        test_folder = get_session_subsample_test_folder(
            subsample_obj.session.session_id,
            subsample_obj.file.filename,
            subsample_id)
        
        train_folder = get_session_subsample_train_folder(
            subsample_obj.session.session_id,
            subsample_obj.file.filename,
            subsample_id)
        
        df_test = read_file(os.path.join(test_folder, os.listdir(test_folder))[0])
        y_test = df_test[target_column]
        X_test = df_test.drop([target_column], axis=1)
        
        for file in os.listdir(train_folder):
            file_path = os.path.join(train_folder, file)
            df_train = read_file(file_path)
            
            y_train = df_train[target_column]
            X_train = df_train.drop([target_column], axis=1)
            
            classifier = dazer.Classifier(X_train, y_train, X_test, y_test)
            
             
            
        
        return Response({})
        



