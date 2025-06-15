#   Raport skryptu do liczenia poruszających się obiektów na scenie drogi
## 1. Dane bazowe i rozeznanie w dostępnych narzędziach
Celem projektu jest zliczanie poruszających się obiektów na scenie drogi, znaleziono bibliotekę [`bgslibrary`](https://github.com/andrewssobral/bgslibrary) z dostępnym wrapperem dla języka Python - `pybgs`, która pozwala na usuwanie tła z obrazu na wiele zaimplementowanych już sposobów. Do jego zbudowania potrzebne było zinstalowanie kilku bibliotek do skompilowania kodu C++ znajdującego się pod komendami python. m.in. `setuptools` i `wheel`. Instalacja przebiegła pomyślnie, ale przetestowane algorytmy nie były zadowalające i najlepsze rezultaty otrzymano po przetworzeniu obrazu [operacjami morfologicznymi](#2-outline-programu), a następnie usuwaniu tła za pomocą różnicy między klatkami plus między pierwszą klatką a kolejnymi - założono korzystną sytuację, że scena jest statyczna, a pierwsza klatka jest tą, na której scena jest w pełni widoczna. Algorytm do śledzenia poruszających się obiektów zaimplementowany w opencv - `cv2.TrackerKCF_create()` - również nie spełniał oczekiwań, mamy dość szczególny przypadek, gdzie drzewo w konkretnym miejscu zasłania drogę. Zdecydowano się na napisanie własnego algorytmu, który będzie śledził poruszające się obiekty.

## 2. Outline programu
Skrypt bazuje na szablonie kodu udostęnionego na platformie eKursy / Github. Skrypt główny pozostał w niezmienionej formie. W tym repozytorium nazwany [`main.py`](../main.py). W głównej pętli funkcji `perform_processing` wykonywane są poniższe kroki:
- Zamiana obrazu na odcienie szarości
    ```python
    grey = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    ```
- Rozmycie obrazu
    ```python
    blurred_frame = cv.GaussianBlur(grey, (9, 9), 0)
    ```
- Ustawienie wartości progującej zdjęcie
    ```python
    th_val = cv.getTrackbarPos('trackbar', 'Camera_Stream_FG')
    ```
- Usunięcie tła z obrazu za pomocą algorytmu - sposób wywoływania algorytmu jest wzorowany na przykładzie z biblioteki `pybgs`. Klasa algorytmu jest opisana w [sekcji 3](#3-algorytm-usuwania-tła). 
    ```python
    fg = algorithm.apply(blurred_frame, threshold=th_val, 
                         update_background=False, two_background=True)
    bg = algorithm.getBackgroundModel()
    ```
Po usunięciu tła, obraz poddano ponownym przekształceniom morfologicznym.
- Operacja otwarcia w celu usunięcia małych obiektów - szumów / liści itp.
    ```python
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (9, 9))
    fg = cv.morphologyEx(fg, cv.MORPH_OPEN, kernel)
    ```
- Operacja zamknięcia w celu usunięcia małych dziur w obiektach
    ```python
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (15, 15))
    fg = cv.morphologyEx(fg, cv.MORPH_CLOSE, kernel)
    ```
- Wykrycie konturów na obrazie za pomocą wbudowanej w bibliotekę funkcję `cv2.findContours`
    ```python
    contours, _ = cv.findContours(fg, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    ```
- Przefiltrowanie konturów, aby pozostały tylko te o powierzchni większej niż 500 pikseli - zabezpieczenie przed nieusuniętymi obiektami takie jak liście itp. i tworzenie obiektów klasy `Bbox` dla każdego konturu, które zawierają informacje o konturze i jego powierzchni
    ```python
    bcontours = [Bbox(cnt) for cnt in contours if cv.contourArea(cnt) > 500]
    ```
- Aktualizacja wykrytych obiektów w trackerze - klasa `CentroidTracker` jest odpowiedzialna za śledzenie poruszających się obiektów na obrazie. Klasa ta jest opisana w [sekcji 5](#5-algorytm-śledzenia-poruszających-się-obiektów).
    ```python
    tracker.update(bcontours)
    ```
Po aktualizacji tracker rysuje prostokąty wokół wykrytych obiektów na obrazie.

<details>
  <summary>Cała funkcja perform_processing()</summary>


```python
def perform_processing(cap: cv.VideoCapture) -> dict[str, int]:
    # cv.namedWindow('Camera_Stream_BG', cv.WINDOW_NORMAL)
    cv.namedWindow('Camera_Stream_FG', cv.WINDOW_NORMAL)
    cv.namedWindow('Camera_Stream_FG_2', cv.WINDOW_NORMAL)
    # cv.namedWindow('frame', cv.WINDOW_NORMAL)
    cv.createTrackbar('trackbar', 'Camera_Stream_FG' , 25, 255, empty)
    tracker_cnt = ContourTracker()
    tracker = CentroidTracker()
    counts = {}
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image.")
            break
        grey = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        blurred_frame = cv.GaussianBlur(grey, (9, 9), 0)
        th_val = cv.getTrackbarPos('trackbar', 'Camera_Stream_FG')

        fg = algorithm.apply(blurred_frame, threshold=th_val, update_background=False, two_background=True)
        bg = algorithm.getBackgroundModel()

        kernel = cv.getStructuringElement(cv.MORPH_RECT, (9, 9))
        fg = cv.morphologyEx(fg, cv.MORPH_OPEN, kernel)
        kernel = cv.getStructuringElement(cv.MORPH_RECT, (15, 15))
        fg = cv.morphologyEx(fg, cv.MORPH_CLOSE, kernel)
        contours, _ = cv.findContours(fg, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        # Split large contours that might represent multiple objects
        # contours = split_contours(contours, fg)
        bcontours = [Bbox(cnt) for cnt in contours if cv.contourArea(cnt) > 500]
        tracker.update(bcontours)
        fg = cv.cvtColor(fg, cv.COLOR_GRAY2BGR)
        tracker.draw(fg)

        frame_counts = tracker.get_counts()
        for key, value in frame_counts.items():
            counts[key] = counts.get(key, 0) + value

        fg2 = fg.copy()
        for bbox in bcontours:
            bbox.draw(fg2)

        cv.imshow('Camera_Stream_FG_2', fg2)
        cv.imshow('Camera_Stream_FG', fg)
        # cv.imshow('frame', frame)
        # cv.imshow('Camera_Stream_BG', bg)
        cv.waitKey(1)

    return {
        "liczba_samochodow_osobowych_z_prawej_na_lewa": counts.get('R2L_car', 0),
        "liczba_samochodow_osobowych_z_lewej_na_prawa": counts.get('L2R_car', 0),
        "liczba_samochodow_ciezarowych_autobusow_z_prawej_na_lewa": counts.get('R2L_truck', 0),
        "liczba_samochodow_ciezarowych_autobusow_z_lewej_na_prawa": counts.get('L2R_truck', 0),
        "liczba_tramwajow": counts.get('TRAM', 0),
        "liczba_pieszych": counts.get('PED', 0),
        "liczba_rowerzystow": counts.get('BIKE', 0)
    }
```

</details>

## 3. Algorytm usuwania tła
Obiekt klasy `FrameDifference` jest odpowiedzialny za usuwanie tła z obrazu. Algorytm ten działa na zasadzie porównywania bieżącej klatki z poprzednią, aby wykryć różnice, które mogą wskazywać na ruch obiektów w scenie. W przypadku pierwszej klatki, jest ona traktowana jako tło, a kolejne klatki są porównywane z nią. Algorytm pozwala na aktualizację tła w trakcie działania programu. Algorytm bazuje na funkcji `cv.absdiff`, która oblicza różnicę między dwoma obrazami, a następnie stosuje prógowanie, aby uzyskać obraz binarny, gdzie białe piksele wskazują na zmiany (ruch) w scenie. Dodatkowo, algorytm pozwala na użycie dwóch różnych modeli tła, co może być przydatne w przypadku dynamicznych scen. Pozwala także zwrócić model tła, który może być użyty do dalszej analizy lub wizualizacji. Istnieje możliwość na rozwój algorytmu, poprzez oddanie funkcji, która aktualizowałaby tło w momencie jej wywołania - funkcja byłaby wywoływana w momencie, gdy scena jest statyczna i nie ma obiektów poruszających się na obrazie. Wtedy algorytm mógłby zaktualizować model tła, co pozwoliłoby na lepsze wykrywanie ruchu w kolejnych klatkach.

<details>
  <summary>Cała klasa algorytmu</summary>


```python
class FrameDifference():
    def __init__(self):
        self.prev_frame_static = None
        self.prev_frame_dynamic = None
        self.mode = 0

    def apply(self, frame, threshold=30, update_background=False, two_background=False):
        if self.prev_frame_static is not None and self.prev_frame_dynamic is not None:
            if update_background:
                diff = cv.absdiff(self.prev_frame_dynamic, frame)
            elif two_background:
                diff1 = cv.absdiff(self.prev_frame_static, frame)
                diff2 = cv.absdiff(self.prev_frame_dynamic, frame)
                diff = cv.bitwise_or(diff1, diff2)
            else:
                diff = cv.absdiff(self.prev_frame_static, frame)
            _, img_output = cv.threshold(diff, threshold, 255, cv.THRESH_BINARY)
        if self.prev_frame_static is None:
            self.prev_frame_static = frame
        self.prev_frame_dynamic = frame
        self.mode = 1 if update_background else 2 if two_background else 3
        return img_output if 'img_output' in locals() else frame
        

    def getBackgroundModel(self):
        if self.prev_frame_static is None or self.prev_frame_dynamic is None:
            return None
        if self.mode == 1:
            return self.prev_frame_dynamic
        elif self.mode == 2:
            return self.prev_frame_static
        elif self.mode == 3:
            return self.prev_frame_static
```

</details>

## 4. Klasy pomocnicze
### Klasa `Bbox`
Klasa `Bbox` jest klasą z której dziedziczy klasa `TrackerObject`. Klasa ta jest odpowiedzialna za przechowywanie informacji o konturze, jego powierzchni, prostokąta otaczającego kontur oraz pozycji kontura - określenie jego typu -  czyli czy jest to samochód osobowy, ciężarowy, tramwaj, pieszy czy rowerzysta. Posiada informacje możliwe do uzyskania z konturu z pojedynczej klatki obrazu. Dla każdej klatki tworzone są nowe obiekty tej klasy, które są następnie porównywane z obiektami z poprzednich klatek za pomocą klasy `CentroidTracker`. Przydałyby się informacje o tym czy kontur jest kwadratowy w każdym krytycznym miejscu - np. ciężarówka z przodu jest kwadratowa, albo każdy pojazd powinien być kwadratowy od spodu. Do tego służyłaby metoda `_is_front_square`, która jest jeszcze do zaimplementowania. 

<details>
  <summary>Cała klasa Bbox</summary>

```python
class Bbox():
    def __init__(self, cnt):
        self.cnt = cnt
        self.x, self.y, self.w, self.h = cv.boundingRect(cnt)
        M = cv.moments(cnt)
        if M['m00'] != 0:
            self.cx = int(M['m10'] / M['m00'])
            self.cy = int(M['m01'] / M['m00'])
        else:
            print("[WARN] Zero moment detected, using bbox center")
            self.cx = self.x + self.w // 2
            self.cy = self.y + self.h // 2
        self.centroid = (self.cx, self.cy)
        self.type = self._find_type()
        self.color = self._assign_color() 
        self.square_front = self._is_front_square()

    def start_corner(self) -> tuple[int, int]:
        return self.x, self.y
    
    def end_corner(self) -> tuple[int, int]:
        return self.x + self.w, self.y + self.h
    
    def _find_type(self) -> Type:
        y_low = self.end_corner()[1]
        if y_low > Row_utils.L2R_LINE:
            return Type.PED
        elif y_low > Row_utils.TRAM_LINE:
            if self.w > self.h * 2:
                return Type.L2R_truck
            else:
                return Type.L2R_car
        elif y_low > Row_utils.R2L_LINE and self.h > 150:
            return Type.TRAM
        else:
            if self.w > self.h * 2:
                return Type.R2L_truck
            else:
                return Type.R2L_car

    def _is_front_square(self) -> bool:
        pass
        
    def _assign_color(self) -> tuple[int, int, int]:
        if self.type == Type.R2L_car or self.type == Type.R2L_truck:
            return Row_utils.R2L_COLOR
        elif self.type == Type.TRAM:
            return Row_utils.TRAM_COLOR
        elif self.type == Type.L2R_car or self.type == Type.L2R_truck:
            return Row_utils.L2R_COLOR
        elif self.type == Type.PED or self.type == Type.BIKE:
            return Row_utils.PED_COLOR

    def draw(self, frame):
        cv.drawContours(frame, [self.cnt], -1, self.color, 2)
        cv.rectangle(frame, self.start_corner(), self.end_corner(), self.color, 2)
        cv.line(
            frame,
            (0, Row_utils.R2L_LINE),
            (frame.shape[1], Row_utils.R2L_LINE),
            Row_utils.TRAM_COLOR,
            2
        )
        cv.line(
            frame,
            (0, Row_utils.TRAM_LINE),
            (frame.shape[1], Row_utils.TRAM_LINE),
            Row_utils.L2R_COLOR,
            2
        )
        cv.line(
            frame,
            (0, Row_utils.L2R_LINE),
            (frame.shape[1], Row_utils.L2R_LINE),
            Row_utils.PED_COLOR,
            2
        )
```

</details>

### Klasa `TrackerObject`
Klasa `TrackerObject` jest klasą dziedziczącą po klasie `Bbox`, która jest odpowiedzialna za przechowywanie informacji o obiekcie śledzonym, bądź w kolejce do śledzenia. Klasa ta zawiera informacje o konturze z klasy Bbox, a także dodatkowe informacje o obiekcie, takie jak licznik klatek, w których obiekt był widoczny, licznik klatek, w których obiekt nie był widoczny, strona , z której obiekt wchodzi na scenę, prędkość obiektu, oraz strefy w których obiekt może być niewidoczny. Obiekty tej klasy są tworzone dla każdego konturu wykrytego na obrazie, a następnie porównywane z obiektami z poprzednich klatek, jeżeli nie ma potrzeby dodania nowego obiektu do śledzenia, to obiekt jest aktualizowany. Klasa ta jest używana przez klasę `CentroidTracker`, która jest odpowiedzialna za kontrolę nad tymi obiektami i wybór czynności dla tego obiektu - 
- `predict()` zakłada, że obiekt nie został wykryty w bieżącej klatce i przewiduje jego pozycję na podstawie jego prędkości.
- `update()` aktualizuje pozycję obiektu na podstawie nowego najbliższego Bboxa, który jest przekazywany do tej metody.
Funkcja `_find_velocity()` musi uwzględniać strefy, w których obiekt może być niewidoczny, co zostało zaimplementowane.

<details>
  <summary>Cała klasa TrackerObject</summary>

```python
class TrackerObject(Bbox):
    def __init__(self, bbox, missed: int = 0, detected_count: int = 1, blackout_zones=[[1000, 1200], [1400, 1700]]):
        super().__init__(bbox.cnt)
        self.missed = missed
        self.detected_count = detected_count
        self.coming_side = self._find_coming_side()
        self.velocity = 1 if self.coming_side == Side.LEFT else -1
        self.blackout_zones = blackout_zones

    def update(self, bbox):
        self.velocity = self._find_velocity(bbox)
        self.cnt =  bbox.cnt
        self.x, self.y, self.w, self.h = cv.boundingRect(bbox.cnt)
        self.detected_count += 1
        self.missed = 0
        # M = cv.moments(bbox.cnt)
        # if M['m00'] != 0:
        #     self.cx = int(M['m10'] / M['m00'])
        #     self.cy = int(M['m01'] / M['m00'])
        # else:
        #     print("[WARN] Zero moment detected, using bbox center")
        self.cx = self.x + self.w // 2
        self.cy = self.y + self.h // 2
        self.centroid = (self.cx, self.cy)

    def predict(self):
        self.x += self.velocity
        self.cx += self.velocity
        self.centroid = (self.cx, self.cy)
        self.missed += 1

    def _find_velocity(self, new_bbox : Bbox) -> int:
        if new_bbox.x < 5 or \
           -20 < new_bbox.x - self.blackout_zones[0][1] < 5 or \
           -20 < new_bbox.x - self.blackout_zones[1][1] < 5:
            return new_bbox.end_corner()[0] - self.end_corner()[0]
        elif new_bbox.end_corner()[0] > FRAME_SHAPE[1] - 5 or \
             -20 < self.blackout_zones[0][0] - new_bbox.end_corner()[0] < 5 or \
             -20 < self.blackout_zones[1][0] - new_bbox.end_corner()[0] < 5:
            return new_bbox.x - self.x
        else:
            return new_bbox.cx - self.cx

    def _find_coming_side(self) -> Side:
        if self.x + self.w / 2 < FRAME_SHAPE[1] / 2:
            return Side.LEFT
        else:
            return Side.RIGHT
```
</details>


## 5. Algorytm śledzenia poruszających się obiektów
Główną funkcją algorytmu jest `update`, która aktualizuje obiekty śledzone na podstawie wykrytych konturów. Jednak zanim porównamy kontury z obiektami śledzonymi, łączymy kontury które się do tego nadają (są blisko siebie poziomo, lub jeden nad drugim), robi to funkcja `_merge_objects`. Rozwiązywane są problemy używania prędkości do łączenia obiektów, które są blisko siebie. Kolejność działań może być rozszerzona o funkcję `_split_objects`, która dzieli obiekty, które są zbyt duże i mogą zawierać więcej niż jeden obiekt. 

<details>
  <summary>Funkcja _merge_objects algorytmu śledzenia obiektów</summary>


```python
def _merge_objects(self, bboxes: list[Bbox], used_indices: set):
    for bbox1 in bboxes:
        for bbox2 in bboxes:
            if bbox1 != bbox2:
                if (bbox1.x < bbox2.cx < bbox1.end_corner()[0]) and \
                    ((bbox1.y < bbox2.cy < bbox1.end_corner()[1]) or \
                    (abs(bbox1.y - bbox2.end_corner()[1]) < 10)):# and \
                    #abs(bbox1.velocity - bbox2.velocity) < 5)):
                    # Merge contours of t_object and pt_object
                        combined_cnt = np.vstack((bbox1.cnt, bbox2.cnt))
                        bboxes.append(Bbox(combined_cnt))
                        bboxes.remove(bbox1)
                        bboxes.remove(bbox2)
                        break
                if (abs(bbox1.y - bbox2.y) < 10 or abs(bbox1.end_corner()[1] - bbox2.end_corner()[1]) < 10) and \
                    (abs(bbox1.cx - bbox2.cx) < 50 or \
                    (abs(bbox1.cx - bbox2.cx) < 600 and (any(abs(bbox2.x - zone[1]) < 20 for zone in self.blackout_zones)))):
                        combined_cnt = np.vstack((bbox1.cnt, bbox2.cnt))
                        bboxes.append(Bbox(combined_cnt))
                        bboxes.remove(bbox1)
                        bboxes.remove(bbox2)
                        break
```

</details>

Po połączeniu konturów, porównujemy je z obiektami śledzonymi `self.objects`  bądź w kolejce do śledzenia `self.pending_objects`. Obiekty są porównywane na podstawie odległości ich środków od siebie. Funkcje są do siebie bardzo podobne jednak nie znaleziono sposobu by zapisać je w jednej, poniżej funkcja `_match_confirmed_objects`.
<details>
  <summary>Funkcja _match_confirmed_objects algorytmu śledzenia obiektów</summary>


```python
def _match_confirmed_objects(self, bboxes: list[Bbox], used_indices: set):
    # t_objest - tracked object
    for t_object in self.objects:
        dists = [distance.euclidean(t_object.centroid, bbox.centroid) for bbox in bboxes]
        min_dist = min(dists)
        idx = dists.index(min_dist) if dists else -1
        if min_dist < self.max_distance and abs(t_object.h - bboxes[idx].h) < self.height_thresh and idx not in used_indices:
            t_object.update(bboxes[idx])
            # print(f"Object matched with new detection at index {idx}, distance {min_dist}, velocity {t_object.velocity}")
            used_indices.add(idx)
        else:
            t_object.predict()
            if t_object.missed > self.max_missed:
                self.objects.remove(t_object)
```

</details>

Jeżeli obiekty `self.pending_objects` uzyskały ponad `min_frames_detected` klatek wykrycia, to są one dodawane do listy obiektów potwierdzonych `self.objects`. Na koniec dodawane są nowe obiekty, które nie zostały jeszcze dodane do listy obiektów śledzonych.

<details>
  <summary>Funkcja _add_new_objects algorytmu śledzenia obiektów</summary>


```python
    def _add_new_objects(self, bboxes: list[Bbox], used_indices: set):
        for oid, bbox in enumerate(bboxes):
            if oid not in used_indices:
                # Only allow creation of new objects at the edges (first and last 200 pixels)
                if bbox.cx < 200 or bbox.cx > (1900 - 200):
                    self.pending_objects.append(TrackerObject(bbox, missed=0, detected_count=1, blackout_zones=self.blackout_zones))
                    used_indices.add(oid)
```

</details>

Poniżej cała sekwencja funkcji `update`.

<details>
  <summary>Funkcja update algorytmu śledzenia obiektów</summary>


```python
def update(self, bboxes: list[Bbox]):
    used_indices = set()
    # If no bboxes, predict all existing objects
    if not bboxes:
        self._predict_all()
        return

    # Merge objects
    bboxes = self._merge_objects(bboxes, used_indices)


    self._match_confirmed_objects(bboxes, used_indices)
    self._match_pending_objects(bboxes, used_indices)


    for pt_object in self.pending_objects:
        if pt_object.detected_count >= self.min_frames_detected and abs(pt_object.velocity) > 5:
            self.objects.append(pt_object)
            self.pending_objects.remove(pt_object)
            self.next_id += 1

    # Add new objects
    self._add_new_objects(bboxes, used_indices)
```

</details>

## 6. Zapis do pliku
Wyniki działania programu są zapisywane do pliku `results.json` za pomocą funkcji `get_counts()` otrzymujemy słownik z liczbą wykrytych obiektów, które przejechały przez scenę na koniec każdej klatki. Następnie te obiekty są usuwane a słownik na koniec jest zwracany. 
<details>
  <summary>Funkcja get_counts algorytmu śledzenia obiektów</summary>

```python
def get_counts(self):
    counts = {Type.R2L_car.name: 0, Type.R2L_truck.name: 0, Type.TRAM.name: 0, Type.L2R_car.name: 0, Type.L2R_truck.name: 0, Type.PED.name: 0}
    
    for obj in self.objects:
        if (obj.coming_side == Side.LEFT and obj.x > FRAME_SHAPE[1] - 20) or \
            (obj.coming_side == Side.RIGHT and obj.end_corner()[0] < 20):
            counts[obj.type.name] += 1
            self.objects.remove(obj)
            
    return counts
```

</details>

## 7. Możliwości rozwoju
W sprawozdaniu można znaleźć kilka zdań dotyczących niezaimplementowanych funkcji, które mogą poprawić działanie algorytmu. Niestety z powodu ograniczonego czasu nie udało się ich zaimplementować. Poniżej znajduje się lista takich usprawnień:
- **_is_front_square** - funkcja, która sprawdza czy kontur jest kwadratowy w każdym krytycznym miejscu - np. ciężarówka z przodu jest kwadratowa, albo każdy pojazd powinien być kwadratowy od spodu. Inaczej można prościej podzielić kontur na dwa mniejsze, które będą odzwierciedlały dwa obiekty, które są blisko siebie.
- **_split_objects** - funkcja, która dzieli obiekty, które cechują się nieregularnymi kształtami, przez co mogą zawierać więcej niż jeden obiekt.
- zaktualizowanie tła w momencie, gdy scena jest statyczna i nie ma obiektów poruszających się na obrazie. Wtedy algorytm mógłby zaktualizować model tła, co pozwoliłoby na lepsze wykrywanie ruchu w kolejnych klatkach.
- implementacja funkcji `_match_pending_objects` i `_match_confirmed_objects` w jednej funkcji, która będzie porównywała kontury z obiektami śledzonymi i w kolejce do śledzenia - jedna pętla.
- Znalezienie lepszego sposobu na znajdowanie prędkości i uwzględnienie jej w przydzielaniu obiektów do klas.

## 8. Wnioski
Nauczono się implementacji algorytmu w czysty i obiektowy sposób, wzorowano się na implementacji z biblioteki `pybgs`. Napotkano wiele trudności związanych z implementacją algorytmu śledzenia obiektów, jednak udało się znaleźć rozwiązania. Zrozumiano jak działają wrappery python na przykładzie `pybgs`. Nauczono się dostosowywać algorytmy do konkretnych przypadków użycia - w tym przypadku napisanie własnego, co było na tyle czasochłonne, że nie udało się zaimplementować wszystkich funkcji.