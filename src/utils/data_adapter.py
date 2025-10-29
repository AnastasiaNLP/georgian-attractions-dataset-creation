"""
Data Adapter for Georgian Attractions Dataset
==============================================

Адаптер для работы с финальной структурой датасета на HuggingFace.

Финальная структура датасета:
- name (одно поле, не name_ru/name_en)
- description (одно поле)
- language (ru/en)
- category (уже обработано)
- ner (уже обработано)
- location
- image

Этот адаптер позволяет использовать enrichment pipeline для НОВЫХ данных
с классической структурой (name_ru/name_en, description_ru/description_en).
"""

import pandas as pd
from typing import Optional
import logging
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


class DatasetAdapter:
    """
    Адаптер для конвертации между различными форматами данных.
    """

    @staticmethod
    def convert_to_classic_format(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Конвертирует финальный формат (name + language) в классический (name_ru/name_en).

        Args:
            df: DataFrame с колонками [name, description, language, ...]

        Returns:
            Tuple (df_ru, df_en) - два DataFrame для русского и английского

        Example:
            >>> df = load_dataset("AIAnastasia/Georgian-attractions")['train'].to_pandas()
            >>> df_ru, df_en = DatasetAdapter.convert_to_classic_format(df)
        """
        if 'language' not in df.columns:
            raise ValueError("Dataset must have 'language' column")

        # Разделяем по языкам
        df_ru = df[df['language'] == 'ru'].copy()
        df_en = df[df['language'] == 'en'].copy()

        # Переименовываем колонки в классический формат
        df_ru = df_ru.rename(columns={
            'name': 'name_ru',
            'description': 'description_ru'
        })

        df_en = df_en.rename(columns={
            'name': 'name_en',
            'description': 'description_en'
        })

        logger.info(f"Converted dataset: {len(df_ru)} Russian, {len(df_en)} English records")

        return df_ru, df_en

    @staticmethod
    def convert_to_final_format(
        df_ru: pd.DataFrame,
        df_en: pd.DataFrame,
        include_enrichments: bool = True
    ) -> pd.DataFrame:
        """
        Конвертирует классический формат обратно в финальный.

        Args:
            df_ru: DataFrame с русскими данными
            df_en: DataFrame с английскими данными
            include_enrichments: Включить ли enrichment поля (category, ner, tags)

        Returns:
            Объединённый DataFrame в финальном формате

        Example:
            >>> df_final = DatasetAdapter.convert_to_final_format(df_ru, df_en)
        """
        # Подготовка русских записей
        ru_records = []
        for _, row in df_ru.iterrows():
            record = {
                'name': row.get('name_ru', ''),
                'description': row.get('description_ru', ''),
                'language': 'ru'
            }

            if include_enrichments:
                # Добавляем enrichment поля если есть
                if 'category_ru' in row:
                    record['category'] = row['category_ru']
                if 'ner_ru' in row:
                    record['ner'] = row['ner_ru']
                if 'tags_ru' in row:
                    record['tags'] = row['tags_ru']

            # Дополнительные поля
            for field in ['location', 'image', 'id']:
                if field in row and pd.notna(row[field]):
                    record[field] = row[field]

            ru_records.append(record)

        # Подготовка английских записей
        en_records = []
        for _, row in df_en.iterrows():
            record = {
                'name': row.get('name_en', ''),
                'description': row.get('description_en', ''),
                'language': 'en'
            }

            if include_enrichments:
                if 'category_en' in row:
                    record['category'] = row['category_en']
                if 'ner_en' in row:
                    record['ner'] = row['ner_en']
                if 'tags_en' in row:
                    record['tags'] = row['tags_en']

            for field in ['location', 'image', 'id']:
                if field in row and pd.notna(row[field]):
                    record[field] = row[field]

            en_records.append(record)

        # Объединяем
        df_final = pd.DataFrame(ru_records + en_records)

        logger.info(f"Created final format dataset: {len(df_final)} total records")

        return df_final

    @staticmethod
    def prepare_for_enrichment(df: pd.DataFrame) -> pd.DataFrame:
        """
        Подготавливает новые данные для enrichment pipeline.

        Ожидаемая структура входных данных:
        - name_ru, name_en
        - description_ru, description_en

        Args:
            df: DataFrame с новыми данными

        Returns:
            Подготовленный DataFrame

        Example:
            >>> new_data = pd.read_csv('new_attractions.csv')
            >>> prepared = DatasetAdapter.prepare_for_enrichment(new_data)
        """
        required_cols = ['name_ru', 'name_en', 'description_ru', 'description_en']
        missing_cols = [col for col in required_cols if col not in df.columns]

        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        # Очистка и валидация
        df_clean = df.copy()

        # Удаляем пустые записи
        for col in required_cols:
            df_clean = df_clean[df_clean[col].notna()]
            df_clean = df_clean[df_clean[col].str.strip() != '']

        logger.info(f"Prepared {len(df_clean)} records for enrichment")

        return df_clean


# Пример использования
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("=" * 60)
    print("DATA ADAPTER EXAMPLES")
    print("=" * 60)

    # Пример 1: Загрузка финального датасета и конвертация
    print("\nExample 1: Convert final format to classic format")
    print("-" * 60)

    from datasets import load_dataset

    # Загружаем финальный датасет
    dataset = load_dataset("AIAnastasia/Georgian-attractions")
    df = dataset['train'].to_pandas()

    print(f"Original dataset: {len(df)} records")
    print(f"Columns: {df.columns.tolist()}")

    # Конвертируем в классический формат
    df_ru, df_en = DatasetAdapter.convert_to_classic_format(df)

    print(f"\nAfter conversion:")
    print(f"Russian: {len(df_ru)} records")
    print(f"English: {len(df_en)} records")

    # Пример 2: Подготовка новых данных
    print("\n\nExample 2: Prepare new data for enrichment")
    print("-" * 60)

    # Создаём пример новых данных
    new_data = pd.DataFrame({
        'name_ru': ['Новая крепость', 'Старый храм'],
        'name_en': ['New Fortress', 'Old Temple'],
        'description_ru': ['Описание крепости...', 'Описание храма...'],
        'description_en': ['Fortress description...', 'Temple description...']
    })

    print(f"New data: {len(new_data)} records")

    prepared = DatasetAdapter.prepare_for_enrichment(new_data)
    print(f"Prepared for enrichment: {len(prepared)} records")

    print("\n✓ Data adapter examples completed!")