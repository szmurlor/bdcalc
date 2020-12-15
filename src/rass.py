# -*- coding: utf-8 -*-
import json
import os
import logging

import shutil


class RASSData:
    """ Klasa do zarządzania danymi w RASSie

        Konstruktor można zainicjalizować na dwa sposoby
        a) za pomocą ścieżki do folderu z danymi, gdzie może znajdować się plik rassdata.json,
           plik ten zawiera specyfikację lokalizacji plików, które mogą być wykorzystywane przez
           klientów obiektu RASSData

        b) za pomocą uchwytu id do obiektu DataSet w bazie danych rass (można opcjonalnie podać ścieżkę
           do bazy danych sqlite.db (zwykle rass.db).

        c) za pomocą ścieżki do pliku rassdata.json,
           plik ten zawiera specyfikację lokalizacji plików, które mogą być wykorzystywane przez
           klientów obiektu RASSData

        Zasada działania jest następująca:

        a) wywołując funkcję (np. input(fname, subfolder) RASSData sprawdza najpierw czy nie jest
           to plik zmapowany za pomocą rassdata.json lub znajdujący się w bazie danych plików.

           Jeżeli jest to plik zmapowany lub znajduje się w bazie danych to sprawdzane jest czy plik
           ten nie został pobrany do repozytorium roboczego.


    """

    RASS_CONFIG_FILE = "rassdata.json"

    def __init__(self, database_id=None, database=None, rassdata=None, root_folder=None):
        self.log = logging.getLogger("RASSData")
        self.data = {
            "root_folder": "",
            "files": {
                "example.dat": {
                    "source_filename": "/etc/hosts",
                    "type": "input"
                }
            }
        }
        self.rassdata_configfile = None

        if root_folder is not None:
            #if not root_folder.startswith("/"):
            #    raise Exception("root_folder argument for RASSData class must be absolute, it must start with '/'.")

            self.data["root_folder"] = root_folder

            # Spróbuj wczytać plik konfiguracyjny w formacie JSON
            rassdata_config = os.path.join(self.data["root_folder"], RASSData.RASS_CONFIG_FILE)
            if os.path.exists(rassdata_config) and os.path.isfile(rassdata_config):
                with open(rassdata_config) as cfname:
                    self.data.update(json.load(cfname))
                self.rassdata_configfile = rassdata_config

        if database_id is not None and database is not None:
            print("Jeszcze tego nie obsługujemy.")

        if rassdata is not None:
            # Spróbuj wczytać plik konfiguracyjny w formacie JSON
            if os.path.isdir(rassdata):
                self.data.update(json.load(rassdata))
                self.rassdata_configfile = rassdata

    def root_folder(self):
        return self.data["root_folder"]

    def _get_folder(self, ftype, fname=None, subfolder=None):
        if fname is None:
            fname = ""
        f = "%s/%s" % (self.data["root_folder"], ftype)
        if not os.path.isdir(f):
            os.mkdir(f)
        if subfolder is not None:
            f = "%s/%s" % (f, subfolder)
            if not os.path.isdir(f):
                os.mkdir(f)

        return "%s/%s" % (f, fname)

    def input(self, fname=None, subfolder=None, check=True):
        """ Jeżeli plik fname istnieje w folderze rootfolder i podfolderze subfolder,
        to jest bezpośrednio do niego zwracany uchwyt. Jak nie istnieje to plik jest kopiowany na podstawie konfiguracji
        rassdataconfig.

        :param fname: nazwa pliku
        :param subfolder: nazwa podkatalogu w folderze input
        :return: pełna ścieżka do pliku
        """
        file = self._get_folder("input", fname, subfolder)
        
        if check:
            if not os.path.isfile(file):
                if fname in self.data["files"]:
                    sourcefile = self.data["files"][fname]["sourcefile"]
                    if os.path.exists(sourcefile) and os.path.isfile(sourcefile):
                        shutil.copy(sourcefile, file)
                    else:
                        logging.error("The sourcefile %s defined in %s file for %s does not exist." % (sourcefile, self.rassdata_configfile, fname))

            if not os.path.isfile(file) and not os.path.isdir(file):
                logging.error("The file %s does not exist and is not defined in %s file." % (file, self.rassdata_configfile))
                file = "File %s does not exist and is not defined in rassdata.json" % fname

        return file

    def input_exists(self, fname=None, subfolder=None):
        fn = self._get_folder("input", fname, subfolder)

        if (os.path.isfile(fn)):
            return True
        if (os.path.isdir(fn)):
            return True
        return False
        
    def output(self, fname=None, subfolder=None):
        return self._get_folder("output", fname, subfolder)

    def output_exists(self, fname=None, subfolder=None):
        fn=self._get_folder("output", fname, subfolder)
        if (os.path.isfile(fn)):
            return True
        if (os.path.isdir(fn)):
            return True
        return False

    def processing(self, fname=None, subfolder=None):
        return self._get_folder("processing", fname, subfolder)

    def clean_out_data(self):
        folder = self.output()
        if os.path.isdir(folder):
            self.clean_folder(folder)
        else:
            self.log.warning("Trying to clean processing data when the out data folder (%s) does not exist." % folder)

    def clean_in_data(self):
        folder = self.input()
        if os.path.isdir(folder):
            self.clean_folder(folder)
        else:
            self.log.warning("Trying to clean processing data when the in data folder (%s) does not exist." % folder)

    def clean_processing_data(self):
        folder = self.processing()
        if os.path.isdir(folder):
            self.clean_folder(folder)
        else:
            self.log.warning("Trying to clean processing data when the processing data folder (%s) does not exist." % folder)

    def clean_folder(self, folder):
        for the_file in os.listdir(folder):
            file_path = os.path.join(folder, the_file)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                self.log.error(e)

    def _init_folder(self, folder):
        if not os.path.isdir(folder):
            os.mkdir(folder)

        return folder

    def output_init_folder(self, folder):
        return self._init_folder(self.output(folder))
