# 1C_files_similarity
- Калмыков Андрей Сергеевич
- Задача №4
## Запуск
Для запуска сначала загрузить библиотеки с помощью `pip3 install requirements.txt`. После этого можно запустить программу с помощью `python3 main.py "first_folder" "second_folder" score results`. Информацию про аргументы командной строки можно получить с помощью `python3 main.py --help`. Вместо всех слов-аргументов стоит поставить нужные значения (названия папок, десятичную дробь для сходства и файл для результатов)
## Принятые решения
На данный момент программа использует в качестве метрики расстояние Дамерау-Левенштейна (учитываются операции добавления, удаления, замены и транспозиции) между содержимым бинарных файлов.

Из-за того, что вычисление расстояния Дамерау-Левенштейна занимает O(n^2) на достаточно длинных файлах программа работает довольно долго.
Кроме того, для ускорения используются параллельные вычисления, что ускоряет программу за счёт большей нагрузки на процессор
## Применение на практике
Данную программу можно применять в качестве антиплагита в контестах по спортивному программированию или чтобы определять, что, например, никто не вынес на флешке важные данные с предприятия
## Тестирование
Для простоты были сгенерированы пара папок и несколько файлов в них, чтобы можно было оценить работоспособность программы