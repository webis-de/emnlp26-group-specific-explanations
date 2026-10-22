import logging
import os
import shutil
import sqlite3
import time
import sqlite_vec  # type: ignore
from typing import List, Literal, Optional, Union, get_args
import numpy as np
import pandas as pd
from data.scripts import get_all_data
from data.prompts import get_description_prompts
from config_new_prompts import (
    BACKUP_FOLDER,
    DATASET_MAPPING,
    STYLE_VECTOR_SIZE,
    STYLE_EMBEDDING_SIZE,
    SELECTED_KNOWLEDGE_PROMPT_IDS,
    SELECTED_STYLE_PROMPT_IDS
)
from helper import STEERING_TYPE_ENUM, format_time, DATA_TYPE_ENUM

pd.options.mode.copy_on_write = True

logger = logging.getLogger(__name__)
backup_point_types = Literal[
    "style_description",
    "style_sentence",
    "base_vector",
    "sfam_style_vector",
    "lisa_style_vector",
    "steering",
]

# mypy: disable-error-code="arg-type, return-value"

ClusterStatus = Literal[
    "rejected", "accepted", "targeted", "pre_accepted1", "pre_accepted2"
]


class Database:
    """A singleton class to handle the database connection.

    Raises:
        RuntimeError: If the database singleton exists but the create_database method was not called.
        RuntimeError: If the database already exists with different parameters.

    Returns:
        Database: The database singleton instance.
    """

    _instance = None
    db_created = False
    file_name = ""

    def __new__(self, db_name: Union[str, None] = None, skip_creation=False):
        if self._instance is not None:
            if not self._instance.db_created and not skip_creation:
                raise RuntimeError("Database singleton exists but was not created")
            return self._instance
        if db_name is None:
            raise ValueError(
                "No database name provided and database was not yet created"
            )
        self._instance = super(Database, self).__new__(self)
        if os.path.exists(db_name):
            self.new_db = False
        else:
            self.new_db = True
        self.file_name = db_name
       
        self._conn = sqlite3.connect(db_name)
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.execute("PRAGMA busy_timeout = 10000;")
        
        if db_name == ":memory:" and os.path.exists("data.sqlite"):
            logger.info("Loading existing database data.sqlite into in-memory database")
            self.new_db = False
            old_db = sqlite3.connect("data.sqlite")
            timer = time.time()
            old_db.backup(self._conn)
            _time_debugging("backup", time.time() - timer, "load_existing_db")
            old_db.close()
        self._conn.enable_load_extension(True)
        sqlite_vec.load(self._conn)
        self._conn.enable_load_extension(False)
        
        #if not self.new_db:
        #    # check self integrity
        #    timer = time.time()
        #    cursor = self._conn.cursor()
        #    cursor.execute("PRAGMA integrity_check;")
        #    result = cursor.fetchone()
        #    cursor.close()
        #    _time_debugging("read", time.time() - timer, "integrity_check")
        #    if result[0] != "ok":
        #        logger.error(f"Database '{db_name}' is not ok")
        #        raise Exception(f"Database '{db_name}' is not ok")
        
        cursor = self._conn.cursor()        
        
        ### questions
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS questions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                sample_id TEXT UNIQUE NOT NULL,
                data_group TEXT NOT NULL,
                data_subgroup TEXT,
                question TEXT NOT NULL
            ) STRICT
            """
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS questions_sampleId_index ON questions(sample_id)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS answers_dataGroup_index ON questions(data_group)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS answers_dataSubgroup_index ON questions(data_subgroup)"
        )

        ### answers
        #cursor.execute(
        #    """
        #    CREATE TABLE IF NOT EXISTS answers (
        #        id INTEGER PRIMARY KEY AUTOINCREMENT,
        #        question_id INTEGER,
        #        data_type TEXT,--- CHECK (data_type IN ({0})),
        #        answer TEXT,
        #        style_vector_data BLOB DEFAULT NULL CHECK (
        #            style_vector_data IS NULL OR vec_length(style_vector_data) == {1}
        #        ),
        #        style_vector_similarity BLOB DEFAULT NULL CHECK (
        #            style_vector_similarity IS NULL OR vec_length(style_vector_similarity) == {1}
        #        ),
        #        style_vector_sfam BLOB DEFAULT NULL CHECK (
        #            style_vector_sfam IS NULL OR vec_length(style_vector_sfam) == {1}
        #        ),
        #        style_vector_lisa BLOB DEFAULT NULL CHECK (
        #            style_vector_lisa IS NULL OR vec_length(style_vector_lisa) == {1}
        #        ),
        #        style_embedding BLOB DEFAULT NULL CHECK (
        #            style_embedding IS NULL OR vec_length(style_embedding) == {2}
        #        ),
        #        FOREIGN KEY(question_id) REFERENCES questions(id)
        #    ) STRICT
        #    """.format(
        #        ", ".join(f"'{data_type}'" for data_type in get_args(DATA_TYPE_ENUM)),
        #        STYLE_VECTOR_SIZE,
        #        STYLE_EMBEDDING_SIZE,
        #    )
        #)
        
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS answers (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                question_id INTEGER,
                data_type TEXT,--- CHECK (data_type IN ({0})),
                answer TEXT,
                style_vector_data BLOB DEFAULT NULL,
                style_vector_similarity BLOB DEFAULT NULL,
                style_vector_sfam BLOB DEFAULT NULL,
                style_vector_lisa BLOB DEFAULT NULL,
                style_embedding BLOB DEFAULT NULL,
                normal_embedding BLOB DEFAULT NULL,
                FOREIGN KEY(question_id) REFERENCES questions(id)
            ) STRICT
            """.format(
                ", ".join(f"'{data_type}'" for data_type in get_args(DATA_TYPE_ENUM))
            )
        )
        
        
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS answers_questionId_index ON answers(question_id)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS answers_dataType_index ON answers(data_type)"
        )

        ### style_descriptions
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS style_descriptions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                answer_id INTEGER,
                prompt_id INTEGER,
                style_description TEXT,
                FOREIGN KEY(answer_id) REFERENCES answers(id),
                FOREIGN KEY(prompt_id) REFERENCES prompts(id)
            ) STRICT
            """
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS description_answerId_index ON style_descriptions(answer_id)"
        )

        ### style_sentences
        # use vec_distance_L2(), vec_distance_L1(), or vec_distance_cosine() as distance metric
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS style_sentences (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                style_sentence TEXT UNIQUE,
                cluster_id INTEGER DEFAULT -1,
                style_cluster_id INTEGER DEFAULT -1,
                knowledge_cluster_id INTEGER DEFAULT -1,
                embedding BLOB,
                FOREIGN KEY(cluster_id) REFERENCES clusters(id),
                FOREIGN KEY(style_cluster_id) REFERENCES clusters(id),
                FOREIGN KEY(knowledge_cluster_id) REFERENCES clusters(id)
            ) STRICT
            """
        )

        # index on style_sentence?
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS styleSentences_clusterId_index ON style_sentences(cluster_id)"
        )
        try:
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS styleSentences_styleClusterId_index ON style_sentences(style_cluster_id)"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS styleSentences_knowledgeClusterId_index ON style_sentences(knowledge_cluster_id)"
            )
        except:
            print("WARNING: No style_cluster_id and knowledge_cluster_id column.")

        ### join_sentence_description
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS join_sentence_description (
                sentence_id INTEGER,
                description_id INTEGER,
                FOREIGN KEY(sentence_id) REFERENCES style_sentences(id),
                FOREIGN KEY(description_id) REFERENCES style_descriptions(id),
                PRIMARY KEY (sentence_id, description_id)
            ) STRICT
            """
        )

        ### style_sentence_outputs
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS style_sentence_outputs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                description_id INTEGER,
                output TEXT,
                is_processed INTEGER DEFAULT FALSE,
                FOREIGN KEY(description_id) REFERENCES style_descriptions(id)
            ) STRICT
            """
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS sentenceOuput_descriptionId_index ON style_sentence_outputs(description_id)"
        )

        ### clusters
        # status: 0 = rejected, 1-8 = pre_accepted[1-8], 8 = target_prompt_clusters, 10 = accepted
        cursor.execute(
            f"""
            CREATE TABLE IF NOT EXISTS clusters (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                center_id INTEGER,
                status INTEGER DEFAULT {self.map_cluster_status("rejected")} NOT NULL,
                style_vector_index INTEGER UNIQUE DEFAULT NULL
                    CHECK (status >= {self.map_cluster_status("targeted")} OR style_vector_index IS NULL),
                prompt_type TEXT CHECK (prompt_type IN ('Style prompt', 'Knowledge prompt')) NOT NULL,
                FOREIGN KEY(center_id) REFERENCES style_sentences(id)
            ) STRICT
            """
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS clusters_status_index ON clusters(status)"
        )
        cursor.execute(
            f"""
            -- Create a trigger to enforce the maximum of STYLE_VECTOR_SIZE 'accepted' entries
            CREATE TRIGGER IF NOT EXISTS check_accepted_limit
            BEFORE INSERT ON clusters
            WHEN NEW.status >= {self.map_cluster_status("targeted")}
            BEGIN
                -- Check the count of accepted rows
                SELECT CASE
                    WHEN (
                        SELECT COUNT(*) 
                        FROM clusters 
                        WHERE status >= {self.map_cluster_status("targeted")}
                        ) >= {STYLE_VECTOR_SIZE} 
                    THEN
                        RAISE(ABORT, 'Maximum of {STYLE_VECTOR_SIZE} accepted clusters exceeded')
                END;
            END;
            """
        )
        cursor.execute(
            f"""
            -- Trigger to enforce the limit during updates
            CREATE TRIGGER IF NOT EXISTS check_accepted_limit_update
            BEFORE UPDATE OF status ON clusters
            WHEN NEW.status >= {self.map_cluster_status("targeted")}
                AND OLD.status < {self.map_cluster_status("targeted")}
            BEGIN
                -- Check the count of 'accepted' rows
                SELECT CASE
                    WHEN (
                        SELECT COUNT(*) 
                        FROM clusters 
                        WHERE status >= {self.map_cluster_status("targeted")}
                        ) >= {STYLE_VECTOR_SIZE} 
                    THEN
                        RAISE(ABORT, 'Maximum of {STYLE_VECTOR_SIZE} accepted clusters exceeded')
                END;
            END;
            """
        )

        ### prompts
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS prompts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                prompt_template TEXT,
                type TEXT CHECK (type IN ('Open prompt', 'Targeted prompt', 'Knowledge prompt')),
                name TEXT
            ) STRICT
            """
        )


        ### steering questions
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS steering_questions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                question TEXT NOT NULL UNIQUE,
                source TEXT NOT NULL
            ) strict
            """
        )

        ### steering generation
        cursor.execute(
            f"""
            CREATE TABLE IF NOT EXISTS steering_generations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                question_id INTEGER NOT NULL,
                data_group TEXT DEFAULT NULL,
                steering_type TEXT NOT NULL,
                llm_name TEXT NOT NULL,
                answer TEXT NOT NULL,
                prompt TEXT NOT NULL,
                style_vector BLOB DEFAULT NULL,
                style_embedding BLOB DEFAULT NULL,
                normal_embedding BLOB DEFAULT NULL,
                activation_layer TEXT DEFAULT NULL CHECK ((activation_layer IS NULL) <> (steering_type LIKE "activation%")),
                activation_lambda REAL DEFAULT NULL CHECK ((activation_lambda IS NULL) <> (steering_type LIKE "activation%")),
                activation_vector BLOB DEFAULT NULL CHECK ((activation_vector IS NULL) <> (steering_type LIKE "activation%")),
                FOREIGN KEY(question_id) REFERENCES steering_questions(id)
            ) STRICT
            """
        )
        # create indeces for steering_type
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS steering_generations_steeringType_index ON steering_generations(steering_type COLLATE NOCASE)"
        )

        ### activation vectors
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS activation_vectors (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                cluster_id INTEGER NOT NULL,
                llm_name TEXT NOT NULL,
                activation_vector BLOB NOT NULL,
                layer INTEGER NOT NULL CHECK (layer >= 0),
                FOREIGN KEY(cluster_id) REFERENCES clusters(id)
            ) STRICT
            """
        )
        
        ### activation vectors
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS activation_vectors_konen (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                answer_id INTEGER NOT NULL,
                llm_name TEXT NOT NULL,
                activation_vector BLOB NOT NULL,
                layer INTEGER NOT NULL CHECK (layer >= 0)
            ) STRICT
            """
        )

        self._conn.commit()
        cursor.close()
        logger.debug("Database initialized")
        return self._instance

    def __del__(self):
        self._conn.close()

    def insert_data(self, data: pd.DataFrame):
        """Save the data from a DataFrame to the database.

        The DataFrame should have the following columns:
        - sample_id: str
        - data_subgroup: str
        - question: str
        - answer: str
        - data_group: str
        - type: 'sfam_train'| 'sfam_test' | 'lisa_train'| 'lisa_test'
        """
        logger.debug("Inserting data into database")
        questions = data[["sample_id", "data_subgroup", "question", "data_group"]]
        questions = questions.drop_duplicates()
        timer = time.time()
        questions.to_sql("questions", self._conn, if_exists="append", index=False)
        self._conn.commit()
        _time_debugging("insert", time.time() - timer, "insert_data-questions")
        cursor = self._conn.cursor()
        timer = time.time()
        for _, row in data.iterrows():
            question_id = cursor.execute(
                "SELECT id FROM questions WHERE sample_id = ?", (row["sample_id"],)
            ).fetchone()[0]
            cursor.execute(
                "INSERT INTO answers (question_id, answer, data_type) VALUES (?, ?, ?)",
                (question_id, row["answer"], row["type"]),
            )
        self._conn.commit()
        cursor.close()
        _time_debugging("insert", time.time() - timer, "insert_data-answers")

    def create_database(
        self,
    ):
        """Create the database with the given parameters. This imports the data and splits it into train, validation and test sets."""
        assert DATASET_MAPPING is not None

        self.db_created = True
        mapping_df = pd.DataFrame(
            [
                {"group_or_dataset": str(item), **DATASET_MAPPING[item]}
                for item in DATASET_MAPPING.keys()
            ],
        )
        mapping_df = mapping_df.fillna(0)
        mapping_df = mapping_df.convert_dtypes(convert_integer=True)
        
        cursor = self._conn.cursor()
        cursor.execute("""
            SELECT name FROM sqlite_master
            WHERE type='table' AND name='meta';
        """)
        table_exists = cursor.fetchone() is not None
        
        if table_exists:
            existing_mapping = pd.read_sql_query("SELECT * FROM meta", self._conn)
            existing_mapping = existing_mapping.convert_dtypes(convert_integer=True)
            import_data_types = existing_mapping.columns
            if not existing_mapping.empty:
                if existing_mapping.equals(mapping_df):
                    logger.info(
                        "Database already exists with the same parameters, skipping import."
                    )
                    return
                if not existing_mapping.equals(mapping_df[existing_mapping.columns]):
                    logger.error(
                        "Database already exists with different parameters, please delete or move the existing database."
                    )
                    raise RuntimeError(
                        "Database already exists with different parameters, please delete or move the existing database."
                    )
                import_data_types = mapping_df.columns.difference(existing_mapping.columns)
                logger.info(
                    f"Adding data for new data_types {', '.join(import_data_types)}"
                )
        
        import_data_types = mapping_df.columns
        mapping_df.to_sql(
            "meta",
            self._conn,
            if_exists="replace",
            index=False,
        )
        self._conn.commit()

        data = get_all_data()
        data = data[data["type"].isin(import_data_types)]
        self.insert_data(data)

        prompt_templates = get_description_prompts()
        prompt_templates.to_sql("prompts", self._conn, if_exists="append", index=False)
        self._conn.commit()
        if import_data_types.equals(mapping_df.columns):
            # do not delete backups if new data is added to an existing database
            self.clear_backups()

    def backup(self, backup_point: backup_point_types):
        db_name = self.file_name
        if backup_point not in get_args(backup_point_types):
            raise ValueError("Invalid backup_point")

        # check self integrity
        cursor = self._conn.cursor()
        cursor.execute("PRAGMA integrity_check;")
        result = cursor.fetchone()
        cursor.close()
        if result[0] != "ok":
            logger.error(f"Database '{db_name}' is not ok")
            raise Exception(f"Database '{db_name}' is not ok")

        # backup database
        if not os.path.exists(BACKUP_FOLDER):
            os.mkdir(BACKUP_FOLDER)
        backup_id = get_args(backup_point_types).index(backup_point)
        backup_name = f"{BACKUP_FOLDER}{backup_id:02d}_{backup_point}.sqlite"
        if os.path.exists(backup_name):
            logger.error(
                f"Backup {backup_name} already exists, saving it as {backup_name}.old"
            )
            shutil.move(backup_name, f"{backup_name}.old")
        backup_database = sqlite3.connect(backup_name)
        timer = time.time()
        self._conn.backup(backup_database)
        self._conn.commit()
        _time_debugging("backup", time.time() - timer, backup_point)
        backup_database.close()
        logger.info(f"Backup saved to {backup_name}")

    @staticmethod
    def clear_backups(clear_from: Union[backup_point_types, None] = None):
        clear_backup = input(f"Should the backup folder {BACKUP_FOLDER} be cleared?")
        if clear_backup == "yes":
            if clear_from is None:
                shutil.rmtree(BACKUP_FOLDER, ignore_errors=True)
                logger.info("All backups cleared")
                return
            if clear_from not in get_args(backup_point_types):
                raise ValueError("Invalid clear_from")
            clear_id = get_args(backup_point_types).index(clear_from)
            for filename in get_args(backup_point_types)[clear_id:]:
                backup_name = f"{BACKUP_FOLDER}{clear_id:02d}_{filename}.sqlite"
                if os.path.exists(backup_name):
                    os.remove(backup_name)
                    logger.info(f"Backup {backup_name} cleared")
                clear_id += 1

    def count_groups(self, data_types: Union[List[str], None] = None) -> int:
        """Count the number of groups that have asked questions in the database."""
        cursor = self._conn.cursor()
        timer = time.time()
        data_types_query = ""
        if data_types is not None:
            data_types_query = "WHERE data_type IN ({})".format(
                ", ".join([f"'{data_type}'" for data_type in data_types])
            )
        cursor.execute(
            f"""
            SELECT COUNT(DISTINCT data_group)
            FROM questions 
            JOIN answers ON questions.id = answers.question_id
            {data_types_query}
            """
        )
        _time_debugging("read", time.time() - timer, "count_groups")
        count = cursor.fetchone()[0]
        cursor.close()
        return count

    def get_groups(self, data_types: Union[List[str], None] = None) -> List[str]:
        """Get the groups that have asked questions in the database."""
        cursor = self._conn.cursor()
        timer = time.time()
        data_types_query = ""
        if data_types is not None:
            data_types_query = "WHERE data_type IN ({})".format(
                ", ".join([f"'{data_type}'" for data_type in data_types])
            )
        cursor.execute(
            f"""
            SELECT DISTINCT data_group
            FROM questions 
            JOIN answers ON questions.id = answers.question_id
            {data_types_query}
            """
        )
        _time_debugging("read", time.time() - timer, "get_groups")
        groups = [row[0] for row in cursor.fetchall()]
        cursor.close()
        return groups

    def get_questions_per_group(
        self, num_per_group: int, group_names: Union[List[str], None] = None
    ) -> pd.DataFrame:
        timer = time.time()
        group_names_query = ""
        if group_names is not None:
            group_names_query = "WHERE data_group IN ({})".format(
                ", ".join([f"'{group_name}'" for group_name in group_names])
            )
        df = pd.read_sql_query(
            f"""
            SELECT id, sample_id, data_group, question
            FROM questions
            {group_names_query}
            ORDER BY data_group
            """,
            self._conn,
        )
        _time_debugging("read", time.time() - timer, "get_questions_per_group")
        # alternativly, the questions could be ordered by how close the answer embedding is to the group embedding
        # when the answer is more representative for the group, this may mean that the question is better
        df = df.groupby("data_group").sample(n=num_per_group, random_state=42)
        return df

    def get_answers(
        self, data_types: Union[None, List[DATA_TYPE_ENUM]] = None
    ) -> pd.DataFrame:
        """Get all answers from the database for the given data types.

        Args:
        - data_types: a list with the types of the data
        """
        if data_types is None:
            data_types = [*get_args(DATA_TYPE_ENUM)]
        if any(data_type not in get_args(DATA_TYPE_ENUM) for data_type in data_types):
            raise ValueError("Invalid data_type")
        timer = time.time()
        df = pd.read_sql_query(
            """
            SELECT answers.id as id, answers.answer, answers.style_vector_data, answers.style_vector_similarity, answers.style_vector_sfam, answers.style_vector_lisa, answers.style_embedding, answers.normal_embedding, answers.data_type, questions.data_group
            FROM answers
            JOIN questions ON answers.question_id = questions.id
            WHERE data_type IN ({});""".format(", ".join(["?" for _ in data_types])),
            self._conn,
            params=data_types,  # type: ignore
        )
        _time_debugging("read", time.time() - timer, "get_answers")
        df["style_vector_data"] = df["style_vector_data"].apply(
            lambda x: np.frombuffer(x, dtype=np.float32) if x is not None else None
        )
        df["style_vector_similarity"] = df["style_vector_similarity"].apply(
            lambda x: np.frombuffer(x, dtype=np.float32) if x is not None else None
        )
        df["style_vector_sfam"] = df["style_vector_sfam"].apply(
            lambda x: np.frombuffer(x, dtype=np.float32) if x is not None else None
        )
        df["style_vector_lisa"] = df["style_vector_lisa"].apply(
            lambda x: np.frombuffer(x, dtype=np.float32) if x is not None else None
        )
        df["style_embedding"] = df["style_embedding"].apply(
            lambda x: np.frombuffer(x, dtype=np.float32) if x is not None else None
        )
        df["normal_embedding"] = df["normal_embedding"].apply(
            lambda x: np.frombuffer(x, dtype=np.float32) if x is not None else None
        )
        return df

    def get_answers_with_sentence_ids(
        self, data_types: List[DATA_TYPE_ENUM]
    ) -> pd.DataFrame:
        """Get all answers from the database for the given data types.

        Args:
        - data_types: a list with the types of the data
        """
        if any(data_type not in get_args(DATA_TYPE_ENUM) for data_type in data_types):
            raise ValueError("Invalid data_type")
        timer = time.time()
        df = pd.read_sql_query(
            """
            SELECT answers.id as id, answers.answer, answers.style_vector_data, answers.style_vector_similarity, answers.style_vector_sfam, answers.style_vector_lisa, answers.data_type, GROUP_CONCAT(style_sentences.id) as sentence_ids
            FROM answers
            JOIN style_descriptions ON answers.id = style_descriptions.answer_id
            JOIN join_sentence_description ON style_descriptions.id = join_sentence_description.description_id
            JOIN style_sentences ON join_sentence_description.sentence_id = style_sentences.id
            WHERE data_type IN ({})
            GROUP BY answers.id
            """.format(", ".join(["?" for _ in data_types])),
            self._conn,
            params=data_types,  # type: ignore
        )
        _time_debugging("read", time.time() - timer, "get_answers")
        df["style_vector_data"] = df["style_vector_data"].apply(
            lambda x: np.frombuffer(x, dtype=np.float32) if x is not None else None
        )
        df["style_vector_similarity"] = df["style_vector_similarity"].apply(
            lambda x: np.frombuffer(x, dtype=np.float32) if x is not None else None
        )
        df["style_vector_sfam"] = df["style_vector_sfam"].apply(
            lambda x: np.frombuffer(x, dtype=np.float32) if x is not None else None
        )
        df["style_vector_lisa"] = df["style_vector_lisa"].apply(
            lambda x: np.frombuffer(x, dtype=np.float32) if x is not None else None
        )
        df["sentence_ids"] = df["sentence_ids"].apply(
            lambda x: np.array([int(i) for i in x.split(",")])
        )
        return df

    def get_answers_without_description(
        self, data_types: List[DATA_TYPE_ENUM]
    ) -> pd.DataFrame:
        """Get all answers from the database that do not have a style description.

        Args:
        - data_types: a list with the types of the data
        """
        timer = time.time()
        df = pd.read_sql_query(
            """
            SELECT answers.id as id, question_id, answer
            FROM answers
            WHERE data_type IN ({})
            AND NOT EXISTS (
                SELECT 1
                FROM style_descriptions
                WHERE style_descriptions.answer_id = answers.id
            )
            """.format(", ".join(["?" for _ in data_types])),
            self._conn,
            params=data_types,  # type: ignore
        )
        _time_debugging("read", time.time() - timer, "get_answers_without_description")
        return df
    
    def get_answers_without_description_of_promptId(
        self, data_types: List[DATA_TYPE_ENUM], prompt_id: int
    ) -> pd.DataFrame:
        """Get all answers from the database that do not have a style description of the given prompt id.

        Args:
        - data_types: a list with the types of the data
        """
        timer = time.time()
        df = pd.read_sql_query(
            """
            SELECT answers.id as id, question_id, answer
            FROM answers
            WHERE data_type IN ({})
            AND NOT EXISTS (
                SELECT 1
                FROM style_descriptions
                WHERE style_descriptions.answer_id = answers.id
                AND style_descriptions.prompt_id = {}
            )
            """.format(", ".join(["?" for _ in data_types]), prompt_id),
            self._conn,
            params=data_types,  # type: ignore
        )
        _time_debugging("read", time.time() - timer, "get_answers_without_description")
        return df

    def count_answers_without_stylevector(
        self,
        vector_type: Literal["data", "similarity", "sfam", "lisa"],
        data_types: List[DATA_TYPE_ENUM],
    ) -> int:
        """Get all answers from the database that do not have a style vector.

        Args:
        - type: the type of the style vector
        """
        if vector_type not in ["data", "similarity", "sfam", "lisa"]:
            raise ValueError("Invalid vector_type")
        if any(data_type not in get_args(DATA_TYPE_ENUM) for data_type in data_types):
            raise ValueError("Invalid data_type")
        data_types_query = " OR ".join(
            [f"data_type = '{data_type}'" for data_type in data_types]
        )
        timer = time.time()
        cursor = self._conn.cursor()
        cursor.execute(
            f"""
            SELECT COUNT(answers.id) as count
            FROM answers
            WHERE style_vector_{vector_type} IS NULL
            AND ({data_types_query})
            """
        )
        _time_debugging(
            "read", time.time() - timer, "count_answers_without_stylevector"
        )
        count = cursor.fetchone()[0]
        cursor.close()
        return count

    def get_answers_with_styleVectorIndex(self, data_types: List[DATA_TYPE_ENUM]):
        def func(x):
            style_vector_df = self.get_style_vector()
            arr = np.zeros(len(style_vector_df), dtype=np.float32)
            np.put(arr, x, 1)
            return arr

        if any(data_type not in get_args(DATA_TYPE_ENUM) for data_type in data_types):
            raise ValueError("Invalid data_type")
        data_types_query = " OR ".join(
            [f"data_type = '{data_type}'" for data_type in data_types]
        )
        timer = time.time()
        df = pd.read_sql_query(
            f"""
            SELECT answers.id as answer_id, clusters.style_vector_index
            FROM answers
            JOIN style_descriptions ON style_descriptions.answer_id = answers.id
            JOIN join_sentence_description ON join_sentence_description.description_id = style_descriptions.id
            JOIN style_sentences ON style_sentences.id = join_sentence_description.sentence_id
            JOIN clusters ON style_sentences.cluster_id = clusters.id
            WHERE ({data_types_query})
            AND clusters.status = {self.map_cluster_status("accepted")}
            GROUP BY answer_id, clusters.style_vector_index
            ORDER BY answer_id, clusters.style_vector_index
            """,
            self._conn,
        )
        _time_debugging(
            "read", time.time() - timer, "get_answers_with_styleVectorIndex"
        )
        df = (
            df.groupby("answer_id")["style_vector_index"]
            .apply(np.hstack)
            .to_frame()
            .reset_index()
        )
        all_answers = self.get_answers(["sfam_train", "sfam_val", "sfam_test"])
        # add all all_answers["id"] that are not in df["answer_id"]
        missing_answer_ids = all_answers[~all_answers["id"].isin(df["answer_id"])][
            "id"
        ].values
        if len(missing_answer_ids) > 0:
            logger.warning(
                f"{len(missing_answer_ids)} answers do not have any style attribute describing them"
            )
            df = pd.concat(
                [
                    df,
                    pd.DataFrame(
                        {
                            "answer_id": missing_answer_ids,
                            "style_vector_index": [np.array([], dtype=np.int32)]
                            * len(missing_answer_ids),
                        }
                    ),
                ],
                ignore_index=True,
            )
        df["style_vector_index"] = df["style_vector_index"].apply(
            lambda x: np.array([], dtype=np.int32)
            if not isinstance(x, np.ndarray) and np.isnan(x)
            else x
        )
        df["style_vector"] = df["style_vector_index"].apply(lambda x: func(x))
        return df

    def count_answers_with_styleVectorIndex(
        self, data_types: List[DATA_TYPE_ENUM] = ["sfam_train", "sfam_val", "sfam_test"]
    ):
        if any(data_type not in get_args(DATA_TYPE_ENUM) for data_type in data_types):
            raise ValueError("Invalid data_type")
        data_types_query = " OR ".join(
            [f"data_type = '{data_type}'" for data_type in data_types]
        )
        timer = time.time()
        df = pd.read_sql_query(
            f"""
            SELECT style_vector_index, COUNT(DISTINCT answers.id) as num_answers
            FROM clusters
            JOIN style_sentences ON clusters.id = style_sentences.cluster_id
            JOIN join_sentence_description ON style_sentences.id = join_sentence_description.sentence_id
            JOIN style_descriptions ON join_sentence_description.description_id = style_descriptions.id
            JOIN answers ON style_descriptions.answer_id = answers.id
            WHERE ({data_types_query})
            AND clusters.status = {self.map_cluster_status("accepted")}
            GROUP BY style_vector_index
            ORDER BY style_vector_index
            """,
            self._conn,
        )
        _time_debugging(
            "read", time.time() - timer, "count_answers_with_styleVectorIndex"
        )
        return df

    def get_answers_with_clusterId_list(
        self,
        data_types: List[DATA_TYPE_ENUM],
        cluster_status: ClusterStatus,
    ) -> pd.DataFrame:
        num_preaccepted2 = self.count_accepted_clusters("pre_accepted2")

        def func(x):
            arr = np.zeros(num_preaccepted2, dtype=np.int32)
            np.put(arr, x, 1)
            return arr

        if any(data_type not in get_args(DATA_TYPE_ENUM) for data_type in data_types):
            raise ValueError("Invalid data_type")
        data_types_query = " OR ".join(
            [f"data_type = '{data_type}'" for data_type in data_types]
        )
        timer = time.time()
        df = pd.read_sql_query(
            f"""
            SELECT answers.id as answer_id, style_sentences.cluster_id
            FROM answers
            JOIN style_descriptions ON style_descriptions.answer_id = answers.id
            JOIN join_sentence_description ON join_sentence_description.description_id = style_descriptions.id
            JOIN style_sentences ON style_sentences.id = join_sentence_description.sentence_id
            JOIN clusters ON style_sentences.cluster_id = clusters.id
            WHERE ({data_types_query})
            AND clusters.status >= {self.map_cluster_status(cluster_status)}
            GROUP BY answer_id, style_sentences.cluster_id
            ORDER BY answer_id, style_sentences.cluster_id
            """,
            self._conn,
        )
        _time_debugging("read", time.time() - timer, "get_answers_with_clusterId_list")
        cluster_id_map = {
            cluster_id: i for i, cluster_id in enumerate(df["cluster_id"].unique())
        }
        df["cluster_id"] = df["cluster_id"].apply(lambda x: cluster_id_map[x])
        df = (
            df.groupby("answer_id")["cluster_id"]
            .apply(np.hstack)
            .to_frame()
            .reset_index()
        )
        all_answers = self.get_answers(["sfam_train", "sfam_val", "sfam_test"])
        # add all all_answers["id"] that are not in df["answer_id"]
        missing_answer_ids = all_answers[~all_answers["id"].isin(df["answer_id"])]["id"]
        if not missing_answer_ids.empty:
            df = pd.concat(
                [
                    df,
                    pd.DataFrame(
                        {
                            "answer_id": missing_answer_ids,
                            "cluster_id": pd.Series(
                                [[] for _ in range(len(missing_answer_ids))]
                            ),
                        }
                    ),
                ],
                ignore_index=True,
            )
        df["cluster_id_list"] = df["cluster_id"].apply(
            lambda x: np.array([], dtype=np.int32)
            if not isinstance(x, np.ndarray) and np.isnan(x)
            else x
        )
        df["cluster_id_data"] = df["cluster_id"].apply(lambda x: func(x))
        df.drop(columns=["cluster_id"], inplace=True)
        return df

    def update_answer_styleVector(
        self,
        data: pd.DataFrame,
        vector_type: Literal["data", "similarity", "sfam", "lisa"],
        skip_logging: bool = False,
    ):
        if vector_type not in ["data", "similarity", "sfam", "lisa"]:
            raise ValueError("Invalid vector_type")
        cursor = self._conn.cursor()
        timer = time.time()
        cursor.executemany(
            f"UPDATE answers SET style_vector_{vector_type} = vec_f32(?) WHERE id = ?",
            data[["style_vector", "answer_id"]].values.tolist(),
        )
        self._conn.commit()
        print(cursor.rowcount)
        if not skip_logging:
            _time_debugging("update", time.time() - timer, "update_answer_styleVector")
        cursor.close()

    def update_answer_styleEmbedding(
        self,
        data: pd.DataFrame,
    ):
        cursor = self._conn.cursor()
        timer = time.time()
        cursor.executemany(
            "UPDATE answers SET style_embedding = vec_f32(?) WHERE id = ?",
            data[["style_embedding", "answer_id"]].values.tolist(),
        )
        self._conn.commit()
        print(cursor.rowcount)
        _time_debugging("update", time.time() - timer, "update_answer_styleEmbedding")
        cursor.close()
        
    def update_answer_normalEmbedding(
        self,
        data: pd.DataFrame,
    ):
        cursor = self._conn.cursor()
        timer = time.time()
        cursor.executemany(
            "UPDATE answers SET normal_embedding = vec_f32(?) WHERE id = ?",
            data[["normal_embedding", "answer_id"]].values.tolist(),
        )
        self._conn.commit()
        _time_debugging("update", time.time() - timer, "update_answer_normalEmbedding")
        cursor.close()

    def remove_style_vector(
        self, vector_type: Literal["data", "similarity", "sfam", "lisa"]
    ):
        if vector_type not in ["data", "similarity", "sfam", "lisa"]:
            raise ValueError("Invalid vector_type")
        cursor = self._conn.cursor()
        timer = time.time()
        cursor.execute(f"UPDATE answers SET style_vector_{vector_type} = NULL")
        self._conn.commit()
        _time_debugging("update", time.time() - timer, "remove_style_vector")
        cursor.close()

    def remove_style_embedding(self):
        cursor = self._conn.cursor()
        timer = time.time()
        cursor.execute("UPDATE answers SET style_embedding = NULL")
        self._conn.commit()
        _time_debugging("update", time.time() - timer, "remove_style_embedding")
        cursor.close()

    def get_style_descriptions_with_sentenceOutputCount(self) -> pd.DataFrame:
        """Get all style descriptions from the database with the count of sentence outputs.

        Returns:
        - pd.DataFrame
        """
        timer = time.time()
        df = pd.read_sql_query(
            """
            SELECT style_descriptions.id, answer_id, style_description, COUNT(output) as output_count, prompt_id
            FROM style_descriptions
            LEFT JOIN style_sentence_outputs ON style_descriptions.id = style_sentence_outputs.description_id
            GROUP BY style_descriptions.id
            """,
            self._conn,
        )
        _time_debugging(
            "read",
            time.time() - timer,
            "get_style_descriptions_with_sentenceOutputCount",
        )
        return df

    def get_style_descriptions_by_answerId(self, answer_id: int) -> pd.DataFrame:
        """Get the style descriptions for a given answer_id.

        Args:
        - answer_id: the id of the answer
        """
        timer = time.time()
        df = pd.read_sql_query(
            "SELECT * FROM style_descriptions WHERE answer_id = ?",
            self._conn,
            params=(int(answer_id),),
        )
        _time_debugging(
            "read", time.time() - timer, "get_style_descriptions_by_answerId"
        )
        return df

    def insert_style_descriptions(self, descriptions: pd.DataFrame):
        """Insert style descriptions into the database.

        Args:
        - descriptions: a DataFrame with the columns 'answer_id', 'prompt_id', 'style_description'
        """
        timer = time.time()
        descriptions.to_sql(
            "style_descriptions", self._conn, if_exists="append", index=False
        )
        self._conn.commit()
        _time_debugging("insert", time.time() - timer, "insert_style_descriptions")

    def insert_style_sentence_outputs(self, outputs: pd.DataFrame):
        """Insert style sentence outputs into the database.

        Args:
        - outputs: a DataFrame with the columns 'description_id', 'output'
        """
        timer = time.time()
        outputs.to_sql(
            "style_sentence_outputs", self._conn, if_exists="append", index=False
        )
        self._conn.commit()
        _time_debugging("insert", time.time() - timer, "insert_style_sentence_outputs")

        
    def get_style_sentence_outputs(self) -> pd.DataFrame:
        """Get all style sentence outputs from the database.

        Returns:
        - pd.DataFrame
        """
        timer = time.time()
        df = pd.read_sql_query(
            """
            SELECT style_sentence_outputs.id,
                style_sentence_outputs.description_id,
                style_sentence_outputs.output,
                style_descriptions.style_description,
                prompts.name as prompt_name,
                prompts.type as prompt_type
            FROM style_sentence_outputs
            JOIN style_descriptions ON style_sentence_outputs.description_id = style_descriptions.id
            JOIN prompts ON style_descriptions.prompt_id = prompts.id
            """,
            self._conn,
        )
        _time_debugging(
            "read", time.time() - timer, "get_style_sentence_outputs"
        )
        return df
        
    def get_style_sentence_outputs_missing_sentences(self) -> pd.DataFrame:
        """Get all style sentence outputs from the database that don't have style sentences extracted from them.

        Returns:
        - pd.DataFrame
        """
        timer = time.time()
        df = pd.read_sql_query(
            """
            SELECT style_sentence_outputs.id,
                style_sentence_outputs.description_id,
                style_sentence_outputs.output,
                style_descriptions.style_description,
                prompts.name as prompt_name,
                prompts.type as prompt_type
            FROM style_sentence_outputs
            JOIN style_descriptions ON style_sentence_outputs.description_id = style_descriptions.id
            JOIN prompts ON style_descriptions.prompt_id = prompts.id
            WHERE style_sentence_outputs.is_processed = 0
            """,
            self._conn,
        )
        _time_debugging(
            "read", time.time() - timer, "get_style_sentence_outputs_missing_sentences"
        )
        return df

    def processed_sentence_outputs(self, output_ids: List[int]):
        cursor = self._conn.cursor()
        timer = time.time()
        cursor.executemany(
            "UPDATE style_sentence_outputs SET is_processed = 1 WHERE id = ?",
            [(output_id,) for output_id in output_ids],
        )
        self._conn.commit()
        _time_debugging("update", time.time() - timer, "processed_sentence_outputs")
        cursor.close()

    def insert_multiple_style_sentences(self, data: pd.DataFrame):
        """Insert multiple style sentences into the database.

        Args:
        - data: a DataFrame with the columns 'description_id', 'style_sentence', 'style_sentence' and 'embedding'
        """
        if data.empty:
            return
        insert_data = data[["style_sentence", "embedding"]]
        insert_data = insert_data.drop_duplicates(subset=["style_sentence"])
        insert_data["embedding"] = insert_data["embedding"].apply(lambda x: x.tobytes())
        timer = time.time()
        insert_data.to_sql(
            "style_sentences", self._conn, if_exists="append", index=False
        )
        self._conn.commit()
        _time_debugging(
            "insert",
            time.time() - timer,
            "insert_multiple_style_sentences-style_sentences",
        )
        cursor = self._conn.cursor()
        join_data = data[["description_id", "style_sentence"]]
        timer = time.time()
        # this could be improved by reading all ids and merging on the sentence
        join_data["sentence_id"] = join_data["style_sentence"].apply(
            lambda x: cursor.execute(
                "SELECT id FROM style_sentences WHERE style_sentence = ?", (x,)
            ).fetchone()[0]
        )
        self._conn.commit()
        _time_debugging(
            "read", time.time() - timer, "insert_multiple_style_sentences-read"
        )
        cursor.close()
        join_data.drop(columns=["style_sentence"], inplace=True)
        timer = time.time()
        join_data.to_sql(
            "join_sentence_description", self._conn, if_exists="append", index=False
        )
        self._conn.commit()
        _time_debugging(
            "insert",
            time.time() - timer,
            "insert_multiple_style_sentences-join_sentence_description",
        )

    
    def get_style_sentences(self, description_id: int) -> pd.DataFrame:
        """Get the style sentences for a given description_id.

        Args:
        - description_id: the id of the style description
        """
        timer = time.time()
        df = pd.read_sql_query(
            """
            SELECT style_sentence
            FROM style_sentences
            JOIN join_sentence_description ON style_sentences.id = join_sentence_description.sentence_id
            WHERE description_id = ?
            """,
            self._conn,
            params=(int(description_id),),
        )
        _time_debugging("read", time.time() - timer, "get_style_sentences")
        return df

    def get_all_style_sentences_and_cluster_ids(self,
    ) -> pd.DataFrame:
        
        timer = time.time()
        df = pd.read_sql_query(
            """
            SELECT id, style_sentence, cluster_id, style_cluster_id, knowledge_cluster_id
            FROM style_sentences
            """,
            self._conn
        )
        _time_debugging("read", time.time() - timer, "get_all_style_sentences_and_cluster_ids")
        return df
    
    def get_all_style_sentences(
        self, data_types: Union[None, List[DATA_TYPE_ENUM]] = None
    ) -> pd.DataFrame:
        if data_types is None:
            data_types = [*get_args(DATA_TYPE_ENUM)]
        current_data_types: List[DATA_TYPE_ENUM] = data_types
        if any(
            data_type not in get_args(DATA_TYPE_ENUM)
            for data_type in current_data_types
        ):
            raise ValueError("Invalid data_type")
        timer = time.time()
        df = pd.read_sql_query(
            """
            SELECT style_sentences.style_sentence, style_sentences.id, style_sentences.embedding
            FROM style_sentences
            JOIN join_sentence_description ON style_sentences.id = join_sentence_description.sentence_id
            JOIN style_descriptions ON join_sentence_description.description_id = style_descriptions.id
            JOIN answers ON style_descriptions.answer_id = answers.id
            WHERE data_type IN ({})
            GROUP BY style_sentences.id
            """.format(", ".join(["?" for _ in current_data_types])),
            self._conn,
            params=current_data_types,
        )
        _time_debugging("read", time.time() - timer, "get_all_style_sentences")
        df["embedding"] = df["embedding"].apply(
            lambda x: np.frombuffer(x, dtype=np.float32)
        )
        return df
    
    def get_all_style_sentences_of_accepted_clusters(
        self, data_types: Union[None, List[DATA_TYPE_ENUM]] = None
    ) -> pd.DataFrame:
        if data_types is None:
            data_types = [*get_args(DATA_TYPE_ENUM)]
        current_data_types: List[DATA_TYPE_ENUM] = data_types
        if any(
            data_type not in get_args(DATA_TYPE_ENUM)
            for data_type in current_data_types
        ):
            raise ValueError("Invalid data_type")
        timer = time.time()
        df = pd.read_sql_query(
            """
            SELECT style_sentences.style_sentence, style_sentences.id, style_sentences.embedding
            FROM style_sentences
            JOIN join_sentence_description ON style_sentences.id = join_sentence_description.sentence_id
            JOIN style_descriptions ON join_sentence_description.description_id = style_descriptions.id
            JOIN answers ON style_descriptions.answer_id = answers.id
            JOIN clusters ON clusters.id = style_sentences.cluster_id
            WHERE data_type IN ({}) AND clusters.style_vector_index IS NOT NULL
            GROUP BY style_sentences.id
            """.format(", ".join(["?" for _ in current_data_types])),
            self._conn,
            params=current_data_types,
        )
        _time_debugging("read", time.time() - timer, "get_all_style_sentences")
        df["embedding"] = df["embedding"].apply(
            lambda x: np.frombuffer(x, dtype=np.float32)
        )
        return df

    def count_style_sentences_missing_cluster(
        self,
        prompt_type: Union[
            List[Literal["Open prompt", "Targeted prompt", "Knowledge prompt"]], None
        ] = None,
    ) -> int:
        """Count the number of style sentences that have not been assigned to a cluster."""
        prompt_type_query = ""
        if prompt_type is not None:
            if any(
                prompt not in ["Open prompt", "Targeted prompt", "Knowledge prompt"]
                for prompt in prompt_type
            ):
                raise ValueError(f"Invalid prompt_type '{prompt_type}'")
            prompt_type_query = (
                "AND ("
                + " OR ".join([f"prompts.type = '{prompt}'" for prompt in prompt_type])
                + ")"
            )
        cursor = self._conn.cursor()
        timer = time.time()
        cursor.execute(
            f"""
            SELECT COUNT(DISTINCT style_sentences.id)
            FROM style_sentences
            JOIN join_sentence_description ON style_sentences.id = join_sentence_description.sentence_id
            JOIN style_descriptions ON join_sentence_description.description_id = style_descriptions.id
            JOIN prompts ON style_descriptions.prompt_id = prompts.id
            WHERE (knowledge_cluster_id = -1 AND style_cluster_id = -1)
            {prompt_type_query}
            """
        )
        count = cursor.fetchone()[0]
        _time_debugging(
            "read", time.time() - timer, "count_style_sentences_missing_cluster"
        )
        cursor.close()
        return count

    #def get_style_sentences_and_occurences(
    #    self,
    #    prompt_type: List[
    #        Literal["Open prompt", "Targeted prompt", "Knowledge prompt"]
    #    ] = [],
    #) -> pd.DataFrame:
    #    """Get all style sentences and the number of times they are used in descriptions. Ordered descending by count."""
    #    if any(
    #        prompt not in ["Open prompt", "Targeted prompt", "Knowledge prompt"]
    #        for prompt in prompt_type
    #    ):
    #        raise ValueError(f"Invalid prompt_type '{prompt_type}'")
    #    
    #    prompt_type_query = ""
    #    if len(prompt_type) > 0:
    #        prompt_type_query = "WHERE " + "(" + " OR ".join(
    #            [f"prompts.type = '{prompt}'" for prompt in prompt_type]
    #        ) + ")"
#
    #        all_prompt_ids = SELECTED_STYLE_PROMPT_IDS + SELECTED_KNOWLEDGE_PROMPT_IDS
    #        id_list_str = ", ".join(str(pid) for pid in all_prompt_ids)
    #        prompt_id_query = f"AND prompts.id IN ({id_list_str})"
    #    else:
    #        all_prompt_ids = SELECTED_STYLE_PROMPT_IDS + SELECTED_KNOWLEDGE_PROMPT_IDS
    #        id_list_str = ", ".join(str(pid) for pid in all_prompt_ids)
    #        prompt_id_query = f"WHERE prompts.id IN ({id_list_str})"
    #    
    #    timer = time.time()
    #    df = pd.read_sql_query(
    #        f"""
    #        SELECT style_sentences.id, style_sentence, COUNT(DISTINCT style_descriptions.id) as count, embedding
    #        FROM style_sentences
    #        JOIN join_sentence_description ON style_sentences.id = join_sentence_description.sentence_id
    #        JOIN style_descriptions ON join_sentence_description.description_id = style_descriptions.id
    #        JOIN prompts ON style_descriptions.prompt_id = prompts.id
    #        {prompt_type_query}
    #        {prompt_id_query}
    #        GROUP BY style_sentences.id
    #        ORDER BY count DESC
    #        """,
    #        self._conn,
    #    )
    #    _time_debugging(
    #        "read", time.time() - timer, "get_style_sentences_and_occurences"
    #    )
    #    df["embedding"] = df["embedding"].apply(
    #        lambda x: np.frombuffer(x, dtype=np.float32)
    #    )
    #    return df
    
    def get_style_sentences_and_prompt_id(
        self,
    ) -> pd.DataFrame:
        """Get all style sentences and the id of the prompt"""
        
        df = pd.read_sql_query(
            f"""
            SELECT style_sentences.id, style_sentence, style_descriptions.id as description_id, prompts.id as prompt_id
            FROM style_sentences
            JOIN join_sentence_description ON style_sentences.id = join_sentence_description.sentence_id
            JOIN style_descriptions ON join_sentence_description.description_id = style_descriptions.id
            JOIN prompts ON style_descriptions.prompt_id = prompts.id
            """,
            self._conn,
        )
        return df
    
    def get_style_sentences_and_occurences(
        self,
        prompt_type: List[
            Literal["Open prompt", "Targeted prompt", "Knowledge prompt"]
        ] = [],
    ) -> pd.DataFrame:
        """Get all style sentences and the number of times they are used in descriptions. Ordered descending by count."""
        if any(
            prompt not in ["Open prompt", "Targeted prompt", "Knowledge prompt"]
            for prompt in prompt_type
        ):
            raise ValueError(f"Invalid prompt_type '{prompt_type}'")
        prompt_type_query = ""
        if len(prompt_type) > 0:
            prompt_type_query = "WHERE " + " OR ".join(
                [f"prompts.type = '{prompt}'" for prompt in prompt_type]
            )
        timer = time.time()
        
        df = pd.read_sql_query(
            f"""
            SELECT style_sentences.id, style_sentence, COUNT(DISTINCT style_descriptions.id) as count, embedding, prompts.id as prompt_id
            FROM style_sentences
            JOIN join_sentence_description ON style_sentences.id = join_sentence_description.sentence_id
            JOIN style_descriptions ON join_sentence_description.description_id = style_descriptions.id
            JOIN prompts ON style_descriptions.prompt_id = prompts.id
            {prompt_type_query}
            GROUP BY style_sentences.id
            ORDER BY count DESC
            """,
            self._conn,
        )
        _time_debugging(
            "read", time.time() - timer, "get_style_sentences_and_occurences"
        )
        df["embedding"] = df["embedding"].apply(
            lambda x: np.frombuffer(x, dtype=np.float32)
        )
        return df


    def insert_clusters(
        self,
        center_ids: List[int],
        prompt_type: Literal["Style prompt", "Knowledge prompt"],
    ):
        """Insert clusters into the database.

        Args:
        - center_ids: a list with the ids of the center sentences for each cluster
        """
        cursor = self._conn.cursor()
        timer = time.time()
        cursor.executemany(
            "INSERT INTO clusters (center_id, prompt_type) VALUES (?, ?)",
            [(center_id, prompt_type) for center_id in center_ids],
        )
        self._conn.commit()
        _time_debugging("insert", time.time() - timer, "insert_clusters")
        cursor.close()

        
    def update_clusters_prompt_type(
        self,
        cluster_ids: List[int], 
        cluster_prompt_type: Literal["Knowledge prompt", "Style prompt"]
    ):
        """Update prompt type of clusters of the database.

        Args:
        - cluster_ids: a list with the id for each cluster
        """
        cursor = self._conn.cursor()
        timer = time.time()
        cursor.executemany(
            "UPDATE clusters SET prompt_type = ? WHERE id = ?",
            [(cluster_prompt_type, cluster_id) for cluster_id in cluster_ids],
        )
        self._conn.commit()
        _time_debugging("insert", time.time() - timer, "update_clusters_prompt_type")
        cursor.close()

    @staticmethod
    def map_cluster_status(
        status: Literal["all", ClusterStatus] = "all",
    ) -> int:
        if status == "all":
            return -1
        elif status == "rejected":
            return 0
        elif status == "pre_accepted1":
            return 1
        elif status == "pre_accepted2":
            return 2
        elif status == "targeted":
            return 9
        elif status == "accepted":
            return 10
        else:
            raise ValueError(f"Invalid status {status}")

    def get_clusters(
        self,
        status: Literal["all", ClusterStatus] = "all",
        include_later=True,
        cluster_type: Literal["all", "style", "knowledge"] = "all",
    ) -> pd.DataFrame:
        """Get all clusters from the database together with the center sentence, the cluster size and the occurence in style_descriptions.

        Returns:
        - pd.DataFrame
        """
        if cluster_type not in ["all", "style", "knowledge"]:
            raise ValueError(f"Invalid cluster_type '{cluster_type}'")
        cluster_type_query = ""
        if cluster_type == "style":
            cluster_type_query = "AND clusters.prompt_type = 'Style prompt'"
        elif cluster_type == "knowledge":
            cluster_type_query = "AND clusters.prompt_type = 'Knowledge prompt'"
            
        timer = time.time()  
        
        df = pd.read_sql_query(
            f"""
            SELECT clusters.id as id, COUNT(DISTINCT style_sentences.id) as size, COUNT(DISTINCT join_sentence_description.description_id) as occurance, style_sentences_center.style_sentence, style_sentences_center.embedding, clusters.style_vector_index, clusters.prompt_type
            FROM clusters
            JOIN style_sentences ON clusters.id = style_sentences.cluster_id
            JOIN join_sentence_description ON style_sentences.id = join_sentence_description.sentence_id
            JOIN style_sentences as style_sentences_center ON clusters.center_id = style_sentences_center.id
            JOIN style_descriptions on join_sentence_description.description_id = style_descriptions.id
            WHERE clusters.status {">=" if include_later else "="} {self.map_cluster_status(status)}
            {cluster_type_query}
            GROUP BY clusters.id
            ORDER BY clusters.style_vector_index ASC
            """,
            self._conn,
        )
        df["embedding"] = df["embedding"].apply(
            lambda x: np.frombuffer(x, dtype=np.float32)
        )
        _time_debugging("read", time.time() - timer, "get_clusters")
        return df

    def get_cluster(self, cluster_id: int, include_rejected=False) -> pd.DataFrame:
        """Get the cluster from the database together with the center sentence, the cluster size and the occurence in style_descriptions.

        Args:
        - cluster_id: the id of the cluster

        Returns:
        - pd.DataFrame
        """
        timer = time.time()
        df = pd.read_sql_query(
            f"""
            SELECT clusters.id as id, COUNT(DISTINCT style_sentences.id) as size, COUNT(DISTINCT join_sentence_description.description_id) as occurance, style_sentence, embedding
            FROM clusters
            JOIN style_sentences ON clusters.center_id = style_sentences.id
            JOIN join_sentence_description ON style_sentences.id = join_sentence_description.sentence_id
            WHERE clusters.id = ?
            AND clusters.status >= {self.map_cluster_status("all" if include_rejected else "accepted")}
            GROUP BY clusters.id
            """,
            self._conn,
            params=(int(cluster_id),),
        )
        df["embedding"] = df["embedding"].apply(
            lambda x: np.fromstring(x[1:-1], sep=", ", dtype=np.float32)
            if x is not None
            else None
        )
        _time_debugging("read", time.time() - timer, "get_cluster")
        return df

    def update_style_sentence_clusters(self, df: pd.DataFrame, cluster_type: str):
        """Update the cluster_id for style sentences in the database.

        Args:
        - df: a DataFrame with the columns 'id' (id of the sentence to update) and 'cluster_id'
        """
        cursor = self._conn.cursor()
        timer = time.time()
        cursor.executemany(
            f"UPDATE style_sentences SET {cluster_type} = ? WHERE id = ?",
            df[[cluster_type, "id"]].values.tolist(),
        )
        self._conn.commit()
        _time_debugging("update", time.time() - timer, "update_style_sentence_clusters")
        cursor.close()

    def get_clusters_with_answers_groups(
        self, cluster_status: ClusterStatus, include_later=True
    ) -> pd.DataFrame:
        """Get all clusters with the number of answers and groups they contain."""
        timer = time.time()
        # select all clusters with at least one answer in sfam_train and sfam_test each
        df = pd.read_sql_query(
            f"""
            SELECT clusters.id, COUNT(DISTINCT answers.id) as answer_count, COUNT(DISTINCT questions.data_group) as group_count
            FROM clusters
            JOIN style_sentences ON clusters.id = style_sentences.cluster_id
            JOIN join_sentence_description ON style_sentences.id = join_sentence_description.sentence_id
            JOIN style_descriptions ON join_sentence_description.description_id = style_descriptions.id
            JOIN answers ON style_descriptions.answer_id = answers.id
            JOIN questions ON answers.question_id = questions.id
            WHERE clusters.status {">=" if include_later else "="} {self.map_cluster_status(cluster_status)}
            GROUP BY clusters.id
            ORDER BY clusters.id
            """,
            self._conn,
        )
        _time_debugging("read", time.time() - timer, "get_clusters_with_answers_groups")
        return df

    def get_clusters_by_group(self, group: str, include_rejected=False) -> pd.DataFrame:
        """Get all clusters from the database together with the center sentence, the cluster size and the occurence in style_descriptions for a given group.

        Args:
        - group: the group of the data

        Returns:
        - pd.DataFrame
        """
        timer = time.time()
        df = pd.read_sql_query(
            f"""
            SELECT clusters.id as cluster_id, COUNT(DISTINCT style_sentences.id) as size, COUNT(DISTINCT join_sentence_description.description_id) as occurance, style_sentences_center.style_sentence, style_sentences_center.embedding
            FROM clusters
            JOIN style_sentences as style_sentences_center ON clusters.center_id = style_sentences_center.id
            JOIN style_sentences ON clusters.id = style_sentences.cluster_id
            JOIN join_sentence_description ON style_sentences.id = join_sentence_description.sentence_id
            JOIN style_descriptions ON join_sentence_description.description_id = style_descriptions.id
            JOIN answers ON style_descriptions.answer_id = answers.id
            JOIN questions ON answers.question_id = questions.id
            WHERE questions.data_group = ?
            AND clusters.status >= {self.map_cluster_status("all" if include_rejected else "accepted")}
            GROUP BY clusters.id
            """,
            self._conn,
            params=(group,),
        )
        df["embedding"] = df["embedding"].apply(
            lambda x: np.frombuffer(x, dtype=np.float32)
        )
        _time_debugging("read", time.time() - timer, "get_clusters_by_group")
        return df

    def pre_accept1_clusters(self, min_groups: int, max_groups: int, min_answers: int):
        """Accept clusters that meet the criteria.

        Args:
        - min_groups: the minimum number of groups a cluster should contain
        - max_groups: the maximum number of groups a cluster should contain
        - min_answers: the minimum number of answers a cluster should contain
        """
        cursor = self._conn.cursor()
        timer = time.time()
        cursor.execute(
            f"""
            UPDATE clusters
            SET status = {self.map_cluster_status("pre_accepted1")}
            WHERE id IN (
                SELECT clusters.id
                FROM clusters
                JOIN style_sentences ON clusters.id = style_sentences.cluster_id
                JOIN join_sentence_description ON style_sentences.id = join_sentence_description.sentence_id
                JOIN style_descriptions ON join_sentence_description.description_id = style_descriptions.id
                JOIN answers ON style_descriptions.answer_id = answers.id
                JOIN questions ON answers.question_id = questions.id
                WHERE clusters.status = {self.map_cluster_status("rejected")}
                GROUP BY clusters.id
                HAVING COUNT(DISTINCT answers.id) >= ?
                AND COUNT(DISTINCT questions.data_group) <= ?
                AND COUNT(DISTINCT questions.data_group) >= ?
            )
            """,
            (min_answers, max_groups, min_groups),
        )
        self._conn.commit()
        _time_debugging("update", time.time() - timer, "pre_accept1_clusters")
        cursor.close()

    def update_cluster_status(
        self,
        cluster_ids: List[int],
        status: ClusterStatus,
    ):
        """Accept clusters on stage 2.

        Args:
        - cluster_ids: the ids of the clusters to accept
        """
        cursor = self._conn.cursor()
        timer = time.time()
        cursor.executemany(
            f"UPDATE clusters SET status = {self.map_cluster_status(status)} WHERE id = ?",
            [(int(cluster_id),) for cluster_id in cluster_ids],
        )
        self._conn.commit()
        _time_debugging("update", time.time() - timer, "update_cluster_status")
        cursor.close()

    def accept_target_clusters(self, data: pd.DataFrame):
        cursor = self._conn.cursor()
        timer = time.time()
        cursor.executemany(
            f"UPDATE clusters SET status = {self.map_cluster_status('targeted')}, style_vector_index = ? WHERE id = ?",
            data[["style_vector_index", "id"]].values.tolist(),
        )
        self._conn.commit()
        _time_debugging("update", time.time() - timer, "accept_target_clusters")
        cursor.close()

    def accept_final_clusters(self, data: pd.DataFrame):
        cursor = self._conn.cursor()
        timer = time.time()
        cursor.executemany(
            f"UPDATE clusters SET status = {self.map_cluster_status('accepted')}, style_vector_index = ? WHERE id = ?",
            data[["style_vector_index", "id"]].values.tolist(),
        )
        self._conn.commit()
        _time_debugging("update", time.time() - timer, "accept_final_clusters")
        cursor.close()

    def count_accepted_clusters(
        self, status: Literal["all", ClusterStatus] = "accepted", include_later=True
    ) -> int:
        """Count the number of accepted clusters."""
        cursor = self._conn.cursor()
        timer = time.time()
        cursor.execute(
            f"""
            SELECT COUNT(*)
            FROM clusters
            WHERE status {">=" if include_later else "="} {self.map_cluster_status(status)}
            """
        )
        count = cursor.fetchone()[0]
        _time_debugging("read", time.time() - timer, "count_accepted_clusters")
        cursor.close()
        return count

    def delete_clusters(self):
        """Delete all clusters and reset the cluster_id in style_sentences."""
        cursor = self._conn.cursor()
        timer = time.time()
        cursor.execute("""
            UPDATE style_sentences
            SET cluster_id = -1,
                style_cluster_id = -1,
                knowledge_cluster_id = -1
        """)
        cursor.execute("DELETE FROM clusters")
        self._conn.commit()
        _time_debugging("delete", time.time() - timer, "delete_clusters")
        cursor.close()
        self.remove_style_vectors()
        self.clear_backups("base_vector")

    def unaccept_clusters(self):
        """Set the status of all clusters to 'rejected'."""
        cursor = self._conn.cursor()
        timer = time.time()
        cursor.execute(
            "UPDATE clusters SET status = ?, style_vector_index = NULL",
            (self.map_cluster_status("rejected"),),
        )
        self._conn.commit()
        _time_debugging("update", time.time() - timer, "unaccept_clusters")
        cursor.close()
        self.remove_style_vectors()
        self.clear_backups("base_vector")

    def remove_style_vectors(self):
        """Remove the style vectors from all answers."""
        cursor = self._conn.cursor()
        timer = time.time()
        cursor.execute(
            "UPDATE answers SET style_vector_data = NULL, style_vector_similarity = NULL, style_vector_sfam = NULL, style_vector_lisa = NULL"
        )
        self._conn.commit()
        _time_debugging("update", time.time() - timer, "remove_style_vectors")
        cursor.close()

    def get_prompt_templates(self):
        """Get all prompt templates from the database."""
        timer = time.time()
        df = pd.read_sql_query("SELECT * FROM prompts", self._conn)
        _time_debugging("read", time.time() - timer, "get_prompt_templates")
        return df

    def get_style_vector(self):
        timer = time.time()
        df = pd.read_sql_query(
            f"""
            SELECT clusters.style_vector_index, style_sentences.id, style_sentences.style_sentence, style_sentences.embedding, clusters.prompt_type
            FROM clusters
            JOIN style_sentences ON clusters.center_id = style_sentences.id
            WHERE clusters.status = {self.map_cluster_status("accepted")}
            ORDER BY clusters.style_vector_index
            """,
            self._conn,
        )
        df["embedding"] = df["embedding"].apply(
            lambda x: np.frombuffer(x, dtype=np.float32)
        )
        _time_debugging("read", time.time() - timer, "get_style_vector")
        return df

    def insert_steering_questions(self, data: pd.DataFrame):
        timer = time.time()
        data.to_sql("steering_questions", self._conn, if_exists="append", index=False)
        self._conn.commit()
        _time_debugging("insert", time.time() - timer, "insert_steering_questions")

    def get_steering_questions(self, split: Literal["val", "test", "all"] = "all"):
        timer = time.time()
        if split == "all":
            df = pd.read_sql_query(f"SELECT * FROM steering_questions", self._conn)
        else:
            df = pd.read_sql_query(f"SELECT * FROM steering_questions WHERE source LIKE '%{split}'", self._conn)
        _time_debugging("read", time.time() - timer, "get_steering_questions")
        return df

    def insert_steering_generation(self, data: pd.DataFrame):
        timer = time.time()
        data.to_sql("steering_generations", self._conn, if_exists="append", index=False)
        self._conn.commit()
        _time_debugging("insert", time.time() - timer, "insert_steering_generation")
    
    def get_all_steering_types_of_llm(
        self, llm_name: str, 
    ):
        timer = time.time()
        df = pd.read_sql_query(
            """
            SELECT DISTINCT steering_type
            FROM steering_generations
            WHERE llm_name = ?
            ORDER BY steering_type
            """,
            self._conn,
            params=(llm_name,),
        )
        _time_debugging("read", time.time() - timer, "get_steering_generation")
        return df
    
    def get_steering_generation_of_llm(
        self, llm_name: str, steering_type: Union[None, STEERING_TYPE_ENUM] = None, 
    ):
        timer = time.time()
        if steering_type is None:
            df = pd.read_sql_query(
                """
                SELECT steering_generations.id, steering_questions.id as question_id, steering_questions.question, steering_type, data_group, llm_name, answer, prompt, style_vector, style_embedding, normal_embedding, activation_layer, activation_lambda ---, activation_vector
                FROM steering_generations
                JOIN steering_questions ON steering_generations.question_id = steering_questions.id
                WHERE llm_name = ?
                ORDER BY data_group
                """,
                self._conn,
                params=(llm_name,),
            )
        else:
            df = pd.read_sql_query(
                """
                SELECT steering_generations.id, steering_questions.id as question_id, steering_questions.question, steering_type, data_group, llm_name, answer, prompt, style_vector, style_embedding, normal_embedding, activation_layer, activation_lambda ---, activation_vector
                FROM steering_generations
                JOIN steering_questions ON steering_generations.question_id = steering_questions.id
                WHERE steering_type = ?
                AND llm_name = ?
                ORDER BY data_group
                """,
                self._conn,
                params=(steering_type,llm_name),
            )
        df["style_vector"] = df["style_vector"].apply(
            lambda x: np.frombuffer(x, dtype=np.float32) if x is not None else None
        )
        df["style_embedding"] = df["style_embedding"].apply(
            lambda x: np.frombuffer(x, dtype=np.float32) if x is not None else None
        )
        df["normal_embedding"] = df["normal_embedding"].apply(
            lambda x: np.frombuffer(x, dtype=np.float32) if x is not None else None
        )
        #df["activation_vector"] = df["activation_vector"].apply(
        #     lambda x: np.frombuffer(x, dtype=np.float32) if x is not None else None
        #)
        _time_debugging("read", time.time() - timer, "get_steering_generation")
        return df
    
    #def get_steering_generation(
    #    self, steering_type: Union[None, STEERING_TYPE_ENUM] = None
    #):
    #    timer = time.time()
    #    if steering_type is None:
    #        df = pd.read_sql_query(
    #            """
    #            SELECT steering_generations.id, steering_questions.question, steering_generations.question_id, steering_type, data_group, llm_name, answer, prompt, style_vector, style_embedding, activation_layer, activation_lambda ---, activation_vector
    #            FROM steering_generations
    #            JOIN steering_questions ON steering_generations.question_id = steering_questions.id
    #            ORDER BY data_group
    #            """,
    #            self._conn,
    #        )
    #    else:
    #        df = pd.read_sql_query(
    #            """
    #            SELECT steering_generations.id, steering_questions.question, steering_type, data_group, llm_name, answer, prompt, style_vector, style_embedding, activation_layer ---, activation_vector
    #            FROM steering_generations
    #            JOIN steering_questions ON steering_generations.question_id = steering_questions.id
    #            WHERE steering_type = ?
    #            ORDER BY data_group
    #            """,
    #            self._conn,
    #            params=(steering_type,),
    #        )
    #    df["style_vector"] = df["style_vector"].apply(
    #        lambda x: np.frombuffer(x, dtype=np.float32) if x is not None else None
    #    )
    #    df["style_embedding"] = df["style_embedding"].apply(
    #        lambda x: np.frombuffer(x, dtype=np.float32) if x is not None else None
    #    )
    #    # df["activation_vector"] = df["activation_vector"].apply(
    #    #     lambda x: np.frombuffer(x, dtype=np.float32) if x is not None else None
    #    # )
    #    _time_debugging("read", time.time() - timer, "get_steering_generation")
    #    return df

    def update_steering_generations_vector(
        self,
        data: pd.DataFrame,
        data_type: Literal["style_vector", "style_embedding", "normal_embedding", "activation_vector"],
    ):
        if data_type not in ["style_vector", "style_embedding", "normal_embedding", "activation_vector"]:
            logger.error(f"unsupported data type {data_type}")
            raise ValueError(f"unsupported data type {data_type}")
        timer = time.time()
        cursor = self._conn.cursor()
        cursor.executemany(
            f"UPDATE steering_generations SET {data_type} = vec_f32(?) WHERE id = ?",
            data[["vector", "id"]].values.tolist(),
        )
        self._conn.commit()
        cursor.close()
        _time_debugging(
            "update", time.time() - timer, "update_steering_generations_vector"
        )

    def remove_steering_generation_of_llm(
        self, llm_name: str, steering_type: Optional[STEERING_TYPE_ENUM] = None
    ):
        cursor = self._conn.cursor()
        timer = time.time()
        if steering_type is None:
            cursor.execute("DELETE FROM steering_generations WHERE llm_name = ?",
                           (llm_name,),
                          )
        else:
            cursor.execute(
                "DELETE FROM steering_generations WHERE steering_type = ? AND llm_name = ?",
                (steering_type,llm_name),
            )
        self._conn.commit()
        _time_debugging("delete", time.time() - timer, "remove_steering_generation")
        cursor.close()
        
    def remove_steering_generation_of_llm_by_question_ids(
        self, llm_name: str, question_ids, steering_type: Optional[STEERING_TYPE_ENUM] = None, 
    ):
        cursor = self._conn.cursor()
        timer = time.time()
        question_ids = ",".join(str(i) for i in question_ids)
        if steering_type is None:
            cursor.execute(f"DELETE FROM steering_generations WHERE llm_name = ? AND question_id IN ({question_ids})",
                           (llm_name,),
                          )
        else:
            cursor.execute(
                f"DELETE FROM steering_generations WHERE steering_type = ? AND llm_name = ? AND question_id IN ({question_ids})",
                (steering_type,llm_name),
            )
        self._conn.commit()
        _time_debugging("delete", time.time() - timer, "remove_steering_generation")
        cursor.close()

    def get_style_vector_missing_activation_vectors(self, llm_name) -> pd.DataFrame:
        timer = time.time()
        df = pd.read_sql_query(
            """
            SELECT clusters.style_vector_index, style_sentences.style_sentence, style_sentences.embedding, clusters.prompt_type
            FROM clusters
            JOIN style_sentences ON clusters.center_id = style_sentences.id
            WHERE 
                clusters.style_vector_index IS NOT NULL
                AND clusters.id NOT IN (
                    SELECT cluster_id
                    FROM activation_vectors
                    WHERE llm_name = ?
                )
            """,
            self._conn,
            params=(llm_name,),
        )
        df["embedding"] = df["embedding"].apply(
            lambda x: np.frombuffer(x, dtype=np.float32)
        )
        _time_debugging(
            "read", time.time() - timer, "get_style_vector_missing_activation_vectors"
        )
        return df
    
    def get_answers_missing_activation_vectors_konen(self, answer_ids, llm_name) -> pd.DataFrame:
        timer = time.time()
        answer_ids = ",".join(str(i) for i in answer_ids)
        df = pd.read_sql_query(
            f"""
            SELECT id, answer
            FROM answers
            WHERE 
                answers.id IN ({answer_ids})
                AND answers.id NOT IN (
                    SELECT answer_id
                    FROM activation_vectors_konen
                    WHERE llm_name = ?
                )
            """,
            self._conn,
            params=(llm_name,),
        )
        _time_debugging(
            "read", time.time() - timer, "get_style_vector_missing_activation_vectors"
        )
        return df

    def insert_activation_vectors(self, data: pd.DataFrame):
        timer = time.time()
        # ensure every vector has dtype float32 and convert to bytes
        data["activation_vector"] = data["activation_vector"].apply(
            lambda x: x.astype(np.float32).tobytes()
        )
        cursor = self._conn.cursor()
        cursor.executemany(
            """INSERT INTO activation_vectors (cluster_id, llm_name, activation_vector, layer) VALUES (
                (
                    SELECT id FROM clusters WHERE style_vector_index = ?
                ), ?, ?, ?
            )
            """,
            data[["style_vector_index", "llm_name", "activation_vector", "layer", ]].values.tolist(),
        )
        self._conn.commit()
        _time_debugging("insert", time.time() - timer, "insert_activation_vector")
        cursor.close()
        
    def insert_activation_vectors_konen(self, data: pd.DataFrame):
        timer = time.time()
        # ensure every vector has dtype float32 and convert to bytes
        data["activation_vector"] = data["activation_vector"].apply(
            lambda x: x.astype(np.float32).tobytes()
        )
        cursor = self._conn.cursor()
        cursor.executemany(
            """INSERT INTO activation_vectors_konen (answer_id, llm_name, activation_vector, layer) VALUES (
                ?, ?, ?, ?
            )
            """,
            data[["answer_id", "llm_name", "activation_vector", "layer", ]].values.tolist(),
        )
        self._conn.commit()
        _time_debugging("insert", time.time() - timer, "insert_activation_vector_konen")
        cursor.close()

    #def get_activation_vectors_by_layer_of_llm(self, layer: int, llm_name: str) -> pd.DataFrame:
    #    timer = time.time()
    #    df = pd.read_sql_query(
    #        """
    #        SELECT clusters.style_vector_index, activation_vectors.llm_name, activation_vectors.activation_vector, activation_vectors.layer
    #        FROM activation_vectors
    #        JOIN clusters ON activation_vectors.cluster_id = clusters.id
    #        WHERE layer = ?
    #        AND llm_name = ?
    #        ORDER BY clusters.style_vector_index ASC
    #        """,
    #        self._conn,
    #        params=(layer,llm_name),
    #    )
    #    df["activation_vector"] = df["activation_vector"].apply(
    #        lambda x: np.frombuffer(x, dtype=np.float32)
    #    )
    #    _time_debugging("read", time.time() - timer, "get_activation_vectors_by_layer")
    #    return df

    def get_all_activation_vectors_of_llm(self, llm_name: str) -> pd.DataFrame:
        timer = time.time()
        df = pd.read_sql_query(
            """
            SELECT clusters.style_vector_index, activation_vectors.llm_name, activation_vectors.activation_vector, activation_vectors.layer
            FROM activation_vectors
            JOIN clusters ON activation_vectors.cluster_id = clusters.id
            WHERE llm_name = ?
            ORDER BY clusters.style_vector_index ASC, activation_vectors.layer ASC
            """,
            self._conn,
            params=(llm_name,)
        )
        df["activation_vector"] = df["activation_vector"].apply(
            lambda x: np.frombuffer(x, dtype=np.float32)
        )
        _time_debugging("read", time.time() - timer, "get_all_activation_vectors")
        return df
    
    def get_all_activation_vectors_konen_of_llm(self, answer_ids, llm_name: str) -> pd.DataFrame:
        timer = time.time()
        answer_ids = ",".join(str(i) for i in answer_ids)
        df = pd.read_sql_query(
            f"""
            SELECT questions.data_group, activation_vectors_konen.answer_id, activation_vectors_konen.llm_name, activation_vectors_konen.activation_vector, activation_vectors_konen.layer
            FROM activation_vectors_konen
            JOIN answers ON activation_vectors_konen.answer_id = answers.id
            JOIN questions ON answers.question_id = questions.id
            WHERE llm_name = ?
            AND answers.id IN ({answer_ids})
            ORDER BY questions.data_group
            """,
            self._conn,
            params=(llm_name,)
        )
        df["activation_vector"] = df["activation_vector"].apply(
            lambda x: np.frombuffer(x, dtype=np.float32)
        )
        _time_debugging("read", time.time() - timer, "get_all_activation_vectors_konen")
        return df

    def delete_activation_vectors(self):
        """Delete all activation vectors."""
        cursor = self._conn.cursor()
        timer = time.time()
        cursor.execute("DELETE FROM activation_vectors")
        self._conn.commit()
        cursor.close()
        _time_debugging("delete", time.time() - timer, "delete_activation_vectors")
        
    def delete_activation_vectors_konen(self):
        """Delete all activation vectors."""
        cursor = self._conn.cursor()
        timer = time.time()
        cursor.execute("DELETE FROM activation_vectors_konen")
        self._conn.commit()
        cursor.close()
        _time_debugging("delete", time.time() - timer, "delete_activation_vectors_konen")

    def get_missing_activation_groups(
        self, group_data_types: List[DATA_TYPE_ENUM]
    ) -> List[str]:
        if any(
            data_type not in get_args(DATA_TYPE_ENUM) for data_type in group_data_types
        ):
            raise ValueError("Invalid data_type")
        timer = time.time()
        cursor = self._conn.cursor()
        cursor.execute(
            f"""
            SELECT DISTINCT data_group
            FROM questions
            JOIN answers ON questions.id = answers.question_id
            WHERE data_type in ({", ".join(["?" for _ in group_data_types])})
            AND data_group NOT IN (
            SELECT data_group
            FROM steering_generations
            WHERE steering_type LIKE "activation_%"
            GROUP BY steering_type, data_group
            HAVING count(question_id) >= (
                (
                SELECT count(*)
                FROM steering_questions
                ) * 2 * 6
            )
            )
            """,
            group_data_types,
        )
        groups = [row[0] for row in cursor.fetchall()]
        cursor.close()
        _time_debugging("read", time.time() - timer, "get_missing_activation_groups")
        return groups


def _time_debugging(
    method: Literal["insert", "update", "delete", "read", "backup"],
    time_s: float,
    caller: str,
):
    logger.debug(
        f"{method} by {caller} took {format_time(time_s)}",
        extra={"method": method, "caller": caller, "time": time_s},
    )
