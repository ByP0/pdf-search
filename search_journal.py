import os
import sys
import re
import pickle
import argparse
import subprocess
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional
import concurrent.futures
import io

# OCR библиотеки
try:
    import fitz  # PyMuPDF
    from PIL import Image
    import pytesseract
    import cv2
    import numpy as np
    OCR_AVAILABLE = True
except ImportError as e:
    print(f"Предупреждение: Не удалось импортировать OCR библиотеки: {e}")
    print("Установите зависимости: pip install PyMuPDF Pillow pytesseract opencv-python numpy")
    OCR_AVAILABLE = False


class JournalSearchEngine:
    def __init__(self, pdfs_folder: Path = Path('local'), cache_folder: Path = Path('ocr_cache')):
        self.pdfs_folder = pdfs_folder
        self.cache_folder = cache_folder
        self.cache_folder.mkdir(exist_ok=True)

        if not OCR_AVAILABLE:
            raise ImportError("OCR библиотеки не установлены. Установите зависимости")

        self.setup_tesseract()

        self.ocr_languages = "rus+eng"

        self.dpi = 200
        self.max_workers = 2 
        
    def setup_tesseract(self):
        """Настройка пути к Tesseract OCR"""
        self.tesseract_available = False

        possible_paths = [
            r"C:\Program Files\Tesseract-OCR\tesseract.exe",
            r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
            "/usr/bin/tesseract",
            "/usr/local/bin/tesseract",
        ]
        
        try:
            if sys.platform == "win32":
                result = subprocess.run(["where", "tesseract"], 
                                      capture_output=True, text=True, shell=True)
            else:
                result = subprocess.run(["which", "tesseract"], 
                                      capture_output=True, text=True)
            
            if result.returncode == 0:
                tesseract_path = result.stdout.strip().split('\n')[0]
                possible_paths.insert(0, tesseract_path)
        except:
            pass

        for tesseract_path in possible_paths:
            if os.path.exists(tesseract_path):
                try:
                    pytesseract.pytesseract.tesseract_cmd = tesseract_path
                    print(f"✓ Найден Tesseract: {tesseract_path}")

                    version = pytesseract.get_tesseract_version()
                    print(f"✓ Версия Tesseract: {version}")
                    
                    self.tesseract_available = True
                    return
                except Exception as e:
                    print(f"✗ Ошибка при проверке {tesseract_path}: {e}")
                    continue
        
        if not self.tesseract_available:
            print("\n" + "="*60)
            print("TESSERACT OCR НЕ НАЙДЕН!")
            print("="*60)
            print("Установите Tesseract OCR с официального сайта")
            print("https://github.com/UB-Mannheim/tesseract/wiki")
            print("="*60)
    
    def load_pdf_files(self) -> List[Path]:

        if not self.pdfs_folder.exists():
            print(f"✗ Папка {self.pdfs_folder} не найдена.")
            return []

        pdf_files = set()

        for ext in ['.pdf', '.PDF', '.Pdf', '.pDF']:
            for file_path in self.pdfs_folder.glob(f'*{ext}'):
                pdf_files.add(file_path)

        unique_files = sorted(pdf_files)
        
        if unique_files:
            print(f"✓ Найдено {len(unique_files)} PDF файлов:")
            for file in unique_files:
                print(f"  - {file.name}")
            return unique_files
        else:
            print(f"✗ PDF файлы не найдены в папке {self.pdfs_folder}")
            return []
        
    def preprocess_image(self, image: Image.Image) -> Image.Image:
        try:
            img_array = np.array(image)
            
            if len(img_array.shape) == 3:
                gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            else:
                gray = img_array
            
            gray = cv2.convertScaleAbs(gray, alpha=1.5, beta=10)
            
            gray = cv2.medianBlur(gray, 1)
            
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
            sharpened = cv2.filter2D(binary, -1, kernel)
            
            return Image.fromarray(sharpened)
            
        except Exception as e:
            print(f"⚠ Ошибка предобработки изображения: {e}")
            return image
    
    def pdf_to_images(self, pdf_path: Path) -> List[Image.Image]:
        images = []
        
        try:
            doc = fitz.open(pdf_path)
            print(f"  Открыт PDF: {pdf_path.name}, страниц: {len(doc)}")
            
            for page_num in range(len(doc)):
                page = doc[page_num]

                zoom = self.dpi / 72 
                mat = fitz.Matrix(zoom, zoom)
             
                pix = page.get_pixmap(matrix=mat, alpha=False)
                
                img_data = pix.tobytes("png")
                image = Image.open(io.BytesIO(img_data))
                images.append(image)
               
                if (page_num + 1) % 10 == 0:
                    print(f"    Конвертировано {page_num + 1} страниц...")
            
            doc.close()
            
        except Exception as e:
            print(f"✗ Ошибка конвертации {pdf_path.name}: {e}")
        
        return images
    
    def extract_text_from_pdf(self, pdf_path: Path, use_cache: bool = True) -> Dict[int, str]:
        if not self.tesseract_available:
            print("✗ Tesseract не доступен")
            return {}
    
        cache_file = self.cache_folder / f"{pdf_path.stem}_cache.pkl"
       
        if use_cache and cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    print(f"✓ Загружаем из кэша: {pdf_path.name}")
                    return pickle.load(f)
            except Exception as e:
                print(f"⚠ Ошибка загрузки кэша {pdf_path.name}: {e}")
        
        print(f"🔍 Обрабатываем файл: {pdf_path.name}")
        page_texts = {}
        
        try:
            images = self.pdf_to_images(pdf_path)
            
            if not images:
                print(f"✗ Не удалось конвертировать {pdf_path.name}")
                return {}
            
            print(f"  Распознаем текст из {len(images)} страниц...")
            
            for page_num, image in enumerate(images, 1):
                processed_image = self.preprocess_image(image)
                
                text = pytesseract.image_to_string(
                    processed_image,
                    lang=self.ocr_languages,
                    config='--psm 3 --oem 3 -c preserve_interword_spaces=1'
                )
                
                page_texts[page_num] = text
                
                if page_num % 5 == 0:
                    print(f"    Распознано {page_num}/{len(images)} страниц...")
            
            print(f"✓ Обработан файл: {pdf_path.name}")
            
        except Exception as e:
            print(f"✗ Ошибка обработки {pdf_path.name}: {e}")
            return {}
        
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(page_texts, f)
            print(f"💾 Сохранено в кэш: {pdf_path.name}")
        except Exception as e:
            print(f"⚠ Ошибка сохранения кэша: {e}")
        
        return page_texts
    
    def search_in_pdf(self, pdf_path: Path, search_words: List[str], 
                      case_sensitive: bool = False, 
                      match_whole_word: bool = True) -> Dict:
        results = {
            'filename': pdf_path.name,
            'filepath': str(pdf_path),
            'found_words': [],
            'total_matches': 0,
            'pages_with_matches': [],
            'details': []
        }
        
        page_texts = self.extract_text_from_pdf(pdf_path)
        
        if not page_texts:
            return results
        
        search_terms = search_words if case_sensitive else [w.lower() for w in search_words]
        
        for page_num, text in page_texts.items():
            search_text = text if case_sensitive else text.lower()
            
            page_matches = []
            
            for i, term in enumerate(search_terms):
                if match_whole_word:
                    pattern = r'\b' + re.escape(term) + r'\b'
                    matches = list(re.finditer(pattern, search_text, re.IGNORECASE))
                else:
                    pattern = re.escape(term)
                    matches = list(re.finditer(pattern, search_text, re.IGNORECASE))
                
                if matches:
                    first_match = matches[0]
                    start_pos = max(0, first_match.start() - 50)
                    end_pos = min(len(text), first_match.end() + 50)
                    context = text[start_pos:end_pos].replace('\n', ' ').strip()
                    
                    if len(context) > 150:
                        context = context[:150] + "..."
                    
                    page_matches.append({
                        'word': search_words[i], 
                        'count': len(matches),
                        'context': context
                    })
            
            if page_matches:
                results['pages_with_matches'].append(page_num)
                results['total_matches'] += sum(m['count'] for m in page_matches)
                results['details'].append({
                    'page': page_num,
                    'matches': page_matches
                })
        
        for detail in results['details']:
            for match in detail['matches']:
                if match['word'] not in results['found_words']:
                    results['found_words'].append(match['word'])
        
        return results
    
    def search_across_all_pdfs(self, search_words: List[str], 
                               max_workers: int = None,
                               **search_kwargs) -> List[Dict]:
        pdf_files = self.load_pdf_files()
        
        if not pdf_files:
            return []
        
        if max_workers is None:
            max_workers = self.max_workers
        
        print(f"\n🔎 Начинаем поиск по {len(pdf_files)} файлам...")
        print(f"📝 Ищем слова: {', '.join(search_words)}")
        print("⏳ Это может занять некоторое время...\n")
        
        results = []
        
        for pdf in pdf_files:
            try:
                result = self.search_in_pdf(pdf, search_words, **search_kwargs)
                if result['total_matches'] > 0:
                    results.append(result)
                    print(f"✅ Найдено в {pdf.name}: {result['total_matches']} совпадений")
                else:
                    print(f"❌ Не найдено в {pdf.name}")
            except Exception as e:
                print(f"⚠ Ошибка при обработке {pdf.name}: {e}")
        
        results.sort(key=lambda x: x['total_matches'], reverse=True)
        
        return results
    
    def print_results(self, results: List[Dict], search_words: List[str]):
        if not results:
            print("\n" + "="*60)
            print("❌ Совпадений не найдено")
            print("="*60)
            return
        
        print("\n" + "="*60)
        print(f"🎉 РЕЗУЛЬТАТЫ ПОИСКА ({len(results)} файлов)")
        print("="*60)
        
        for i, result in enumerate(results, 1):
            print(f"\n{i}. 📄 Файл: {result['filename']}")
            print(f"   🔢 Совпадений: {result['total_matches']}")
            print(f"   📖 Страницы: {', '.join(map(str, result['pages_with_matches']))}")
            
            if result['found_words']:
                print(f"   🏷️  Найденные слова: {', '.join(result['found_words'])}")
            
            if result['details']:
                print("   📝 Контекст:")
                for detail in result['details'][:2]: 
                    print(f"   - Страница {detail['page']}:")
                    for match in detail['matches'][:2]: 
                        print(f"     '{match['word']}': {match['context']}")
        
        print("\n" + "="*60)
        print(f"📊 Всего найдено в {len(results)} файлах")
        print("="*60)


def main():
    parser = argparse.ArgumentParser(description='Поиск по отсканированным PDF журналам')
    parser.add_argument('words', nargs='+', help='Слова для поиска')
    parser.add_argument('--folder', default='local', help='Папка с PDF файлами')
    parser.add_argument('--cache', default='ocr_cache', help='Папка для кэша')
    parser.add_argument('--no-cache', action='store_true', help='Не использовать кэш')
    parser.add_argument('--case-sensitive', action='store_true', help='Учитывать регистр')
    parser.add_argument('--partial', action='store_true', help='Искать частичные совпадения')
    parser.add_argument('--threads', type=int, default=1, help='Количество потоков')
    
    args = parser.parse_args()
    
    try:
        engine = JournalSearchEngine(
            pdfs_folder=Path(args.folder),
            cache_folder=Path(args.cache)
        )
        
        if not args.no_cache:
            print("💾 Используется кэш")
        
        results = engine.search_across_all_pdfs(
            search_words=args.words,
            max_workers=args.threads,
            case_sensitive=args.case_sensitive,
            match_whole_word=not args.partial
        )
        
        engine.print_results(results, args.words)
        
        if results:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"search_results_{timestamp}.txt"
            
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(f"Поиск: {', '.join(args.words)}\n")
                f.write(f"Время: {datetime.now()}\n")
                f.write(f"Найдено файлов: {len(results)}\n\n")
                
                for result in results:
                    f.write(f"{'='*50}\n")
                    f.write(f"Файл: {result['filename']}\n")
                    f.write(f"Совпадений: {result['total_matches']}\n")
                    f.write(f"Страницы: {', '.join(map(str, result['pages_with_matches']))}\n")
                    
                    if result['found_words']:
                        f.write(f"Слова: {', '.join(result['found_words'])}\n")
                    
                    f.write("\n")
            
            print(f"\n💾 Результаты сохранены в {output_file}")
        
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        return 1
    
    return 0


def interactive_search():
    print("="*60)
    print("🔍 ПОИСК ПО ЖУРНАЛАМ (OCR)")
    print("="*60)
    
    try:
        engine = JournalSearchEngine()
        
        while True:
            print("\nВведите слова для поиска (через пробел) или 'q' для выхода:")
            user_input = input("> ").strip()
            
            if user_input.lower() in ['q', 'quit', 'exit', 'выход']:
                print("👋 Выход из программы")
                break
            
            if not user_input:
                continue
            
            search_words = [w.strip() for w in user_input.split() if w.strip()]
            
            print("\nПараметры поиска:")
            print("1. Учитывать регистр")
            print("2. Искать частичные совпадения")
            print("3. Обычный поиск (целые слова, без учета регистра)")
            
            choice = input("Выберите вариант (1-3, по умолчанию 3): ").strip()
            
            case_sensitive = (choice == '1')
            match_whole_word = (choice != '2')
            
            print(f"\n🔎 Начинаю поиск: {', '.join(search_words)}...")
            
            results = engine.search_across_all_pdfs(
                search_words=search_words,
                case_sensitive=case_sensitive,
                match_whole_word=match_whole_word
            )
            
            engine.print_results(results, search_words)
            
            if results:
                save = input("\n💾 Сохранить результаты в файл? (y/n): ").lower()
                if save == 'y':
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    search_str = '_'.join(search_words[:3])  # Берем первые 3 слова
                    output_file = f"search_{search_str}_{timestamp}.txt"
                    
                    with open(output_file, 'w', encoding='utf-8') as f:
                        f.write(f"Поиск: {', '.join(search_words)}\n")
                        f.write(f"Время: {datetime.now()}\n\n")
                        
                        for result in results:
                            f.write(f"{'='*50}\n")
                            f.write(f"Файл: {result['filename']}\n")
                            f.write(f"Совпадений: {result['total_matches']}\n")
                            f.write(f"Страницы: {', '.join(map(str, result['pages_with_matches']))}\n\n")
                            
                            for detail in result['details']:
                                f.write(f"Страница {detail['page']}:\n")
                                for match in detail['matches']:
                                    f.write(f"  - {match['word']}: {match['context']}\n")
                                f.write("\n")
                    
                    print(f"✅ Результаты сохранены в {output_file}")
    
    except KeyboardInterrupt:
        print("\n\n⚠ Программа прервана пользователем")
    except Exception as e:
        print(f"❌ Ошибка: {e}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        sys.exit(main())
    else:
        interactive_search()