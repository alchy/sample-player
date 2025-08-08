#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
code: sample-player.py
MIDI Sampler - Opravená verze s vylepšenou interpolací
=====================================================s
VERSION HASH: 7f8e3a9d-2025-08-05-v2.2-FIXED
Author: Opraveno pro správnou interpolaci velocity a DbLvl
Nástroj pro mapování vzorků na MIDI velocity a generování chybějících not
s pokročilým cachováním a strukturovaným kódem.
"""
import os
import re
import argparse
import logging
import shutil
import resampy
import numpy as np
import torchaudio
import torch
import pygame.midi
import pygame.mixer
from collections import defaultdict
from uuid import uuid4
from typing import Dict, List, Tuple, Optional


class Config:
    """Konfigurace aplikace."""
    VERSION_HASH = "7f8e3a9d-2025-08-05-v2.2-FIXED"
    VELOCITY_LEVELS = 8
    MIDI_VELOCITY_MAX = 127
    TEMP_DIR_NAME = "samples_tmp"
    MAX_PITCH_SHIFT = 12  # Maximální povolený posun noty (v půltónech)
    MIDI_NOTE_RANGE = range(21, 109)


class AudioFile:
    """Reprezentuje jeden audio soubor se všemi jeho metadaty."""

    def __init__(self, filepath: str, midi_note: int, note_name: str, db_level: int):
        self.filepath = filepath
        self.midi_note = midi_note
        self.note_name = note_name
        self.db_level = db_level
        self.filename = os.path.basename(filepath)

    @classmethod
    def from_filename(cls, filepath: str) -> Optional['AudioFile']:
        """Vytvoří AudioFile z cesty k souboru, pokud odpovídá formátu."""
        filename = os.path.basename(filepath)
        # FIXOVANÝ REGEX - nyní s .wav na konci!
        match = re.match(r"m(\d{3})-([A-G]#?_\d)-DbLvl([+-]?\d+)\.wav", filename)
        if match:
            midi_note = int(match.group(1))
            note_name = match.group(2)
            db_level = int(match.group(3))
            return cls(filepath, midi_note, note_name, db_level)
        return None

    def __repr__(self):
        return f"AudioFile(note={self.midi_note}, db={self.db_level}, file={self.filename})"


class VelocityMapper:
    """Stará se o mapování MIDI velocity na audio soubory."""

    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.velocity_map: Dict[Tuple[int, int], str] = {}  # (midi_note, velocity) -> filepath
        self.available_notes: set = set()
        self.note_samples: Dict[int, List[AudioFile]] = defaultdict(list)  # Všechny vzorky pro každou notu

    def build_velocity_map(self, input_dir: str) -> None:
        """Vytvoří mapování MIDI not a velocity na zvukové soubory."""
        self.logger.info("🔍 Prohledávám složku '%s' pro výběr vzorků podle MIDI velocity...", input_dir)
        note_db_map = defaultdict(list)
        # Načtení všech audio souborů
        for filename in os.listdir(input_dir):
            if not filename.endswith(".wav"):
                continue
            filepath = os.path.join(input_dir, filename)
            audio_file = AudioFile.from_filename(filepath)
            if audio_file:
                note_db_map[audio_file.midi_note].append((audio_file.db_level, audio_file.filepath))
                self.available_notes.add(audio_file.midi_note)
                self.note_samples[audio_file.midi_note].append(audio_file)
                self.logger.debug("Načten soubor: nota=%d, DbLvl=%d, soubor=%s",
                                  audio_file.midi_note, audio_file.db_level, filename)
            else:
                self.logger.debug("Přeskočen soubor (neodpovídá formátu): %s", filename)
        # Vytvoření velocity mapování
        for midi_note, db_files in note_db_map.items():
            db_files.sort(key=lambda x: x[0])  # Seřazení podle db_level (od nejnižšího k nejvyššímu)
            total = len(db_files)
            self.logger.info("🎵 MIDI nota %d: %d různých úrovní hlasitosti (DbLvl)", midi_note, total)
            self.logger.debug("DbLvl úrovně pro notu %d: %s", midi_note, [db for db, _ in db_files])
            ranges = self._get_velocity_ranges(total)
            self.logger.debug("Velocity rozsahy pro notu %d: %s", midi_note, ranges)
            for i, ((db_level, filepath), (v_start, v_end)) in enumerate(zip(db_files, ranges)):
                self.logger.debug("DbLvl %d -> velocity rozsah %d-%d (%s)",
                                  db_level, v_start, v_end, os.path.basename(filepath))
                for velocity in range(v_start, v_end + 1):
                    self.velocity_map[(midi_note, velocity)] = filepath
                    if velocity == v_start or velocity == v_end:  # Logujeme jen hranice pro přehlednost
                        self.logger.debug("Mapování: nota=%d, velocity=%d -> %s",
                                          midi_note, velocity, os.path.basename(filepath))
        self.logger.info("✅ Velocity map vytvořen pro %d not.", len(self.available_notes))

    def _get_velocity_ranges(self, level_count: int) -> List[Tuple[int, int]]:
        """Vytvoří rozsahy MIDI velocity na základě počtu úrovní."""
        step = Config.MIDI_VELOCITY_MAX / level_count
        ranges = []
        for i in range(level_count):
            start = round(i * step)
            end = round((i + 1) * step) - 1
            if i == level_count - 1:  # Poslední rozsah jde až do maxima
                end = Config.MIDI_VELOCITY_MAX - 1
            ranges.append((start, end))
        return ranges

    def get_sample_for_velocity(self, midi_note: int, velocity: int) -> Optional[str]:
        """Vrátí cestu k vzorku pro danou notu a velocity."""
        sample = self.velocity_map.get((midi_note, velocity))
        self.logger.debug("Hledám vzorek: nota=%d, velocity=%d -> %s",
                          midi_note, velocity, sample if sample else "nenalezeno")
        return sample

    def add_generated_sample(self, midi_note: int, velocity_range: Tuple[int, int], filepath: str) -> None:
        """Přidá vygenerovaný vzorek do mapování."""
        v_start, v_end = velocity_range
        for velocity in range(v_start, v_end + 1):
            self.velocity_map[(midi_note, velocity)] = filepath
            self.logger.debug("Přidán vygenerovaný vzorek: nota=%d, velocity=%d -> %s",
                              midi_note, velocity, os.path.basename(filepath))

    def get_samples_for_note(self, midi_note: int) -> List[AudioFile]:
        """Vrátí všechny vzorky pro danou notu seřazené podle DbLvl."""
        samples = self.note_samples.get(midi_note, [])
        return sorted(samples, key=lambda x: x.db_level)


class SampleGenerator:
    """Generuje chybějící vzorky pomocí resamplingu s vylepšenou interpolací."""

    def __init__(self, velocity_mapper: VelocityMapper):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.velocity_mapper = velocity_mapper

    def generate_missing_notes(self, temp_dir: str) -> None:
        """Vygeneruje chybějící noty s různými velocity úrovněmi."""
        self.logger.info("🛠️ Předzpracování: kontrola a vytváření chybějících tónů...")
        base_notes = sorted(self.velocity_mapper.available_notes)
        if not base_notes:
            self.logger.error("Žádné základní noty k dispozici pro generování!")
            return
        for note in Config.MIDI_NOTE_RANGE:
            if note in self.velocity_mapper.available_notes:
                self.logger.debug("Nota %d již existuje v původních vzorcích, přeskočena.", note)
                continue
            # Kontrola, zda už existují vygenerované vzorky pro tuto notu
            if self._note_samples_exist_in_temp(note, temp_dir):
                self.logger.info("♻️ Nota %d: používám existující vygenerované vzorky", note)
                self._load_existing_samples_for_note(note, temp_dir)
                continue
            # Generování nové noty s více velocity úrovněmi
            self._generate_note_with_multiple_velocities(note, base_notes, temp_dir)

    def _note_samples_exist_in_temp(self, note: int, temp_dir: str) -> bool:
        """Zkontroluje, zda už existují vzorky pro danou notu v temp složce."""
        if not os.path.exists(temp_dir):
            return False
        pattern = f"m{note:03d}-"
        for filename in os.listdir(temp_dir):
            if filename.startswith(pattern) and filename.endswith(".wav"):
                return True
        return False

    def _load_existing_samples_for_note(self, note: int, temp_dir: str) -> None:
        """Načte existující vzorky pro notu z temp složky do velocity mapy."""
        pattern = f"m{note:03d}-"
        samples_found = []
        for filename in os.listdir(temp_dir):
            if filename.startswith(pattern) and filename.endswith(".wav"):
                filepath = os.path.join(temp_dir, filename)
                # Extrahujeme velocity z názvu souboru (pokud je tam)
                velocity_match = re.search(r"-v(\d+)\.wav$", filename)
                if velocity_match:
                    target_velocity = int(velocity_match.group(1))
                    samples_found.append((target_velocity, filepath))
        # Seskupíme podle velocity do rozsahů
        if samples_found:
            samples_found.sort(key=lambda x: x[0])
            # Vytvoříme rozsahy na základě skutečných velocity hodnot
            velocity_ranges = self._create_ranges_from_velocities([v for v, _ in samples_found])
            for i, (velocity_range, filepath) in enumerate(zip(velocity_ranges, [f for _, f in samples_found])):
                self.velocity_mapper.add_generated_sample(note, velocity_range, filepath)
            self.logger.debug("Načteno %d existujících vzorků pro notu %d s rozsahy: %s",
                              len(samples_found), note,
                              [f"{start}-{end}" for start, end in velocity_ranges])

    def _create_ranges_from_velocities(self, velocities: List[int]) -> List[Tuple[int, int]]:
        """Vytvoří velocity rozsahy z existujících velocity hodnot."""
        if not velocities:
            return []
        if len(velocities) == 1:
            # Jeden vzorek pokryje celý rozsah
            return [(0, Config.MIDI_VELOCITY_MAX - 1)]
        ranges = []
        sorted_velocities = sorted(velocities)
        for i, velocity in enumerate(sorted_velocities):
            if i == 0:
                # První rozsah: od 0 do poloviny mezi prvním a druhým
                end = (velocity + sorted_velocities[i + 1]) // 2 - 1
                ranges.append((0, end))
            elif i == len(sorted_velocities) - 1:
                # Poslední rozsah: od poloviny mezi předposledním a posledním do maxima
                start = (sorted_velocities[i - 1] + velocity) // 2
                ranges.append((start, Config.MIDI_VELOCITY_MAX - 1))
            else:
                # Střední rozsahy
                start = (sorted_velocities[i - 1] + velocity) // 2
                end = (velocity + sorted_velocities[i + 1]) // 2 - 1
                ranges.append((start, end))
        return ranges

    def _generate_note_with_multiple_velocities(self, note: int, base_notes: List[int], temp_dir: str) -> None:
        """Vygeneruje notu s různými velocity úrovněmi interpolovanou z nejbližších not."""
        # Najdeme dvě nejbližší noty pro lepší interpolaci
        lower_note = None
        upper_note = None

        for base_note in base_notes:
            if base_note < note:
                if lower_note is None or base_note > lower_note:
                    lower_note = base_note
            elif base_note > note:
                if upper_note is None or base_note < upper_note:
                    upper_note = base_note
        # Vybereme primární a sekundární zdrojovou notu
        if lower_note is not None and upper_note is not None:
            # Máme noty z obou stran - vybereme bližší jako primární
            if abs(note - lower_note) <= abs(note - upper_note):
                primary_source = lower_note
                secondary_source = upper_note
            else:
                primary_source = upper_note
                secondary_source = lower_note
        elif lower_note is not None:
            primary_source = lower_note
            secondary_source = None
        elif upper_note is not None:
            primary_source = upper_note
            secondary_source = None
        else:
            # Fallback na nejbližší notu
            primary_source = min(base_notes, key=lambda n: abs(n - note))
            secondary_source = None
        if abs(note - primary_source) > Config.MAX_PITCH_SHIFT:
            self.logger.warning("⚠️ Přeskočena nota %d: příliš velký rozdíl od zdroje %d", note, primary_source)
            return
        self.logger.info("🎯 Generuji notu %d z primárního zdroje %d%s",
                         note, primary_source,
                         f" (sekundární: {secondary_source})" if secondary_source else "")
        # Získáme všechny vzorky pro primární zdrojovou notu
        primary_samples = self.velocity_mapper.get_samples_for_note(primary_source)
        if not primary_samples:
            self.logger.warning("Žádné vzorky nalezeny pro primární zdrojovou notu %d", primary_source)
            return
        # Generujeme všechny dostupné velocity úrovně ze zdrojových vzorků
        generated_samples = []
        num_source_samples = len(primary_samples)

        self.logger.info("🔄 Generuji %d velocity úrovní z %d zdrojových vzorků pro notu %d",
                         num_source_samples, num_source_samples, note)

        for i, source_sample in enumerate(primary_samples):
            # Vypočítáme cílovou velocity pro tento vzorek proporcionálně
            velocity_step = Config.MIDI_VELOCITY_MAX / num_source_samples
            target_velocity = int((i + 0.5) * velocity_step)
            target_velocity = min(target_velocity, Config.MIDI_VELOCITY_MAX - 1)

            # Vytvoříme velocity rozsah pro tento vzorek
            if i == 0:
                v_start = 0
            else:
                v_start = int(i * velocity_step)

            if i == num_source_samples - 1:
                v_end = Config.MIDI_VELOCITY_MAX - 1
            else:
                v_end = int((i + 1) * velocity_step) - 1

            velocity_range = (v_start, v_end)
            try:
                generated_path = self._resample_to_note_with_velocity(
                    note, source_sample, primary_source, target_velocity, temp_dir
                )
                if generated_path:
                    generated_samples.append((velocity_range, generated_path))
                    self.logger.info("✅ Vytvořen: %s (ze zdroje: nota=%d, DbLvl=%d -> velocity=%d)",
                                     os.path.basename(generated_path),
                                     primary_source, source_sample.db_level, target_velocity)
            except Exception as e:
                self.logger.error("⚠️ Chyba při generování noty %d z %d (DbLvl %d): %s",
                                  note, primary_source, source_sample.db_level, e)
        # Přidání vygenerovaných vzorků do velocity mapy
        for velocity_range, filepath in generated_samples:
            self.velocity_mapper.add_generated_sample(note, velocity_range, filepath)
        self.logger.info("🎵 Vygenerováno %d velocity úrovní pro notu %d", len(generated_samples), note)

    def _resample_to_note_with_velocity(self, target_note: int, source_sample: AudioFile,
                                        source_note: int, target_velocity: int, temp_dir: str) -> Optional[str]:
        """Resampluje zvukový vzorek na cílovou notu se správnou velocity."""
        self.logger.debug(
            "Resampling: cílová nota=%d, zdrojový soubor=%s, zdrojová nota=%d, DbLvl=%d, cílová velocity=%d",
            target_note, source_sample.filename, source_note, source_sample.db_level, target_velocity)
        try:
            # Načtení a zpracování audia
            y, sr = torchaudio.load(source_sample.filepath)
            y = y.numpy()[0]
            # Výpočet pitch shift ratio
            ratio = 2 ** ((target_note - source_note) / 12.0)
            y_rs = resampy.resample(y, sr, sr / ratio)
            # Úprava amplitudy podle velocity - zachováváme původní DbLvl!
            # Neměníme amplitudu podle velocity, protože DbLvl už reprezentuje hlasitost
            # Velocity slouží pouze pro mapování, ne pro změnu hlasitosti

            # Výpočet správného názvu noty pro cílovou MIDI notu
            target_note_name = self._midi_note_to_name(target_note)

            # Vytvoření názvu souboru se zachovanou původní DbLvl
            out_filename = f"m{target_note:03d}-{target_note_name}-DbLvl{source_sample.db_level}-v{target_velocity}.wav"
            out_path = os.path.join(temp_dir, out_filename)
            # Uložení
            torchaudio.save(out_path, torch.tensor(y_rs).unsqueeze(0), sr)
            self.logger.debug("Pitch shift ratio: %.3f, zachována původní DbLvl: %d",
                              ratio, source_sample.db_level)
            return out_path
        except Exception as e:
            self.logger.error("Chyba při resamplingu noty %d z %s: %s", target_note, source_sample.filepath, e)
            return None

    def _midi_note_to_name(self, midi_note: int) -> str:
        """Převede MIDI číslo noty na název ve formátu NOTE_OCTAVE (např. A_0, Bb_4)."""
        # Názvy not v pořadí od C
        note_names = ['C', 'Db', 'D', 'Eb', 'E', 'F', 'Gb', 'G', 'Ab', 'A', 'Bb', 'B']

        # MIDI nota 60 = C4 (střední C)
        # MIDI nota 21 = A0 (nejnižší A na klavíru)

        # Výpočet oktávy - MIDI nota 21 (A0) je v oktávě 0
        octave = (midi_note - 12) // 12

        # Výpočet pozice noty v rámci oktávy
        note_index = midi_note % 12
        note_name = note_names[note_index]

        # Převod # na b pro konzistenci s existujícím formátem
        if '#' in note_name:
            note_name = note_name.replace('#', 'b')

        return f"{note_name}_{octave}"


class DirectoryManager:
    """Spravuje dočasné složky a soubory."""

    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.script_dir = os.path.dirname(os.path.abspath(__file__))

    def ensure_temp_dir(self, clean: bool = False) -> str:
        """Vytvoří dočasnou složku. Pokud clean=True, vyčistí ji před použitím."""
        temp_dir = os.path.join(self.script_dir, Config.TEMP_DIR_NAME)
        if clean and os.path.exists(temp_dir):
            self.logger.debug("Čištění existující dočasné složky: %s", temp_dir)
            shutil.rmtree(temp_dir)
        os.makedirs(temp_dir, exist_ok=True)
        self.logger.info("📁 Používám dočasnou složku: %s", temp_dir)
        return temp_dir


class MidiPlayer:
    """Zpracovává MIDI vstup a přehrává vzorky."""

    def __init__(self, velocity_mapper: VelocityMapper):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.velocity_mapper = velocity_mapper
        self.midi_in = None

    def start_midi_input_loop(self) -> None:
        """Spustí hlavní smyčku pro zpracování MIDI vstupu."""
        pygame.midi.init()
        pygame.mixer.init()
        try:
            device_id = self._select_midi_device()
            self.midi_in = pygame.midi.Input(device_id)
            self.logger.info("🎼 Poslouchám MIDI vstup... Ukončete stisknutím Ctrl+C.")
            self._process_midi_events()
        except KeyboardInterrupt:
            self.logger.info("🛑 Ukončeno uživatelem.")
        finally:
            self._cleanup()

    def _select_midi_device(self) -> int:
        """Zobrazí dostupná MIDI zařízení a nechá uživatele vybrat."""
        input_count = pygame.midi.get_count()
        self.logger.info("Dostupné MIDI vstupy:")
        for i in range(input_count):
            interf, name, input_dev, _, _ = pygame.midi.get_device_info(i)
            if input_dev:
                self.logger.info("%d: %s", i, name.decode())
        try:
            device_id = int(input("Vyberte MIDI vstup podle čísla (nebo stiskněte Enter pro první zařízení): ") or 0)
            self.logger.info("Vybrán MIDI vstup: %d", device_id)
            return device_id
        except ValueError:
            self.logger.info("Výchozí MIDI vstup: 0")
            return 0

    def _process_midi_events(self) -> None:
        """Zpracovává MIDI události v nekonečné smyčce."""
        while True:
            if self.midi_in.poll():
                events = self.midi_in.read(10)
                for event in events:
                    data, _ = event
                    self._handle_midi_event(data)

    def _handle_midi_event(self, data: List[int]) -> None:
        """Zpracuje jednotlivou MIDI událost."""
        # Note On
        if data[0] & 0xF0 == 0x90 and data[2] > 0:
            note = data[1]
            velocity = data[2]
            sample = self.velocity_mapper.get_sample_for_velocity(note, velocity)
            if sample:
                pygame.mixer.Sound(sample).play()
                self.logger.info("▶️ Přehrávám: nota %d, velocity %d -> %s",
                                 note, velocity, os.path.basename(sample))
            else:
                self.logger.warning("⚠️ Vzorek nenalezen: nota %d, velocity %d", note, velocity)
        # Note Off
        elif data[0] & 0xF0 == 0x80 or (data[0] & 0xF0 == 0x90 and data[2] == 0):
            note = data[1]
            self.logger.info("⏹️ Zastavuji: nota %d", note)

    def _cleanup(self) -> None:
        """Vyčistí MIDI a mixer resources."""
        if self.midi_in:
            self.midi_in.close()
        pygame.midi.quit()
        pygame.mixer.quit()
        self.logger.debug("MIDI a mixer ukončeny.")


class MidiSampler:
    """Hlavní třída aplikace, která koordinuje všechny komponenty."""

    def __init__(self, input_dir: str, debug: bool = False, clean_temp: bool = False):
        self.setup_logging(debug)
        self.logger = logging.getLogger(self.__class__.__name__)
        self.input_dir = input_dir
        self.clean_temp = clean_temp
        # Inicializace komponent
        self.directory_manager = DirectoryManager()
        self.velocity_mapper = VelocityMapper()
        self.sample_generator = SampleGenerator(self.velocity_mapper)
        self.midi_player = MidiPlayer(self.velocity_mapper)

    def setup_logging(self, debug_mode: bool) -> None:
        """Nastavení logování s volitelnou úrovní podrobnosti."""
        logging.basicConfig(
            level=logging.DEBUG if debug_mode else logging.INFO,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            handlers=[logging.StreamHandler()]
        )
        if debug_mode:
            logging.getLogger().debug("Logování nastaveno v debug režimu.")

    def run(self) -> None:
        """Spustí celou aplikaci."""
        self.logger.info("🚀 Spouštím MIDI Sampler v%s...", Config.VERSION_HASH)
        # 1. Vytvoření velocity mapy z původních vzorků
        self.velocity_mapper.build_velocity_map(self.input_dir)
        # 2. Příprava dočasné složky
        temp_dir = self.directory_manager.ensure_temp_dir(clean=self.clean_temp)
        # 3. Generování chybějících vzorků (s cachováním)
        self.sample_generator.generate_missing_notes(temp_dir)
        # 4. Spuštění MIDI přehrávání
        self.midi_player.start_midi_input_loop()


def parse_args():
    """Parsuje argumenty příkazové řádky."""
    parser = argparse.ArgumentParser(
        description=f"MIDI Sampler v{Config.VERSION_HASH} - Nástroj pro mapování vzorků na MIDI velocity a generování chybějících not.")
    parser.add_argument('--input-dir', required=True,
                        help='Složka se vzorky pojmenovanými jako mNNN-NOTA-DbLvlX.wav')
    parser.add_argument('--debug', action='store_true',
                        help='Zapne podrobné logování')
    parser.add_argument('--clean-temp', action='store_true',
                        help='Vyčistí dočasnou složku před spuštěním (znovu vygeneruje všechny vzorky)')
    parser.add_argument('--version', action='store_true',
                        help='Zobrazí verzi a ukončí program')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.version:
        print(f"MIDI Sampler Version: {Config.VERSION_HASH}")
        exit(0)
    sampler = MidiSampler(
        input_dir=args.input_dir,
        debug=args.debug,
        clean_temp=args.clean_temp
    )
    sampler.run()
