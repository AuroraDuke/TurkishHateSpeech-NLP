import pandas as pd
import re
import emoji
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# İlk çalıştırmada gerekli NLTK indirmeleri
nltk.download('punkt')
nltk.download('stopwords')

# Türkçe stopwords listesini al
turkce_stopwords = stopwords.words('turkish')

# Excel dosyasının yolunu belirle
file_path = "Türkçe Nefret Söylemi Veriseti.xlsx"

# Excel dosyasını yükle
excel_data = pd.ExcelFile(file_path)

# Tüm sekmeleri oku (başlıkları normalize ederek)
sheets = []
for name in excel_data.sheet_names[1:]:
    sheet = pd.read_excel(file_path, sheet_name=name, header=1)  # Başlık satırı doğru tanımlanıyor.
    
    # Sütun adlarını normalize et
    sheet.columns = [col.strip().lower().replace(" ", "_") for col in sheet.columns]
    
    # Eğer 'rowid' varsa, bunu 'row_id' olarak yeniden adlandır
    sheet.rename(columns={'rowid': 'row_id'}, inplace=True)
    
    # Index'i sıfırla, ilk satırı atma
    sheet.reset_index(drop=True, inplace=True)
    
    # Tüm değerleri boş olan sütunları sil (hem NaN hem boş string için kontrol)
    sheet = sheet.loc[:, ~(sheet.isna().all() | sheet.apply(lambda x: x.str.strip().eq("").all() if x.dtype == "object" else False))]
    
    # Tek bir değere sahip sütunları sil
    sheet = sheet.loc[:, sheet.nunique(dropna=True) > 1]
    
    sheets.append(sheet)

# Sekmeleri birleştir
merged_data = pd.concat(sheets, ignore_index=True)

# Birleştirme sonrası da tümü boş veya tek bir değere sahip sütunları kontrol et ve sil
merged_data = merged_data.loc[:, ~(merged_data.isna().all() | merged_data.apply(lambda x: x.str.strip().eq("").all() if x.dtype == "object" else False))]
merged_data = merged_data.loc[:, merged_data.nunique(dropna=True) > 1]

# Metin temizleme ve işleme fonksiyonu
def temizle_ve_isle(metin):
    if not isinstance(metin, str):  # Eğer hücre metin değilse orijinal veriyi döndür
        return metin
    
    # 1. URL'leri kaldır
    metin = re.sub(r"http\S+|www\S+|https\S+", "", metin, flags=re.MULTILINE)
    
    # 2. Noktalama işaretlerini kaldır
    metin = re.sub(r"[^\w\s]", "", metin)
    
    # 3. Emojileri kaldır
    metin = emoji.replace_emoji(metin, "")
    
    # 4. Büyük/küçük harf duyarlılığını kaldır
    metin = metin.lower()
    
    # 5. Tokenize et (kelimelere ayır)
    kelimeler = word_tokenize(metin)
    
    # 6. Stopwords'leri çıkar
    temiz_kelimeler = [kelime for kelime in kelimeler if kelime not in turkce_stopwords]
    
    # 7. Listeyi birleştirerek geri döndür
    return " ".join(temiz_kelimeler)

# Tüm hücreleri kontrol et ve temizle
for col in merged_data.columns:
    merged_data[col] = merged_data[col].apply(temizle_ve_isle)

# Gereksiz sütunları çıkar
columns_to_drop = [
    'tweet_id', 'time', 'favorited', 'retweeted', 'is_favourited', 'is_retweeted', 
    'is_retweet', 'retweet_from', 'latitude', 'longitude', 'country', 'user', 
    'user_-_profile_image', 'user_-_name', 'user_-_id', 'user_-_description', 
    'user_-_url', 'user_-_creation_time', 'user_-_language', 'user_-_location', 
    'user_-_time_zone', 'user_-_statuses', 'user_-_followers', 'user_-_friends', 
    'user_-_favourites'
]

# merged_data DataFrame'inden belirtilen sütunları sil
merged_data = merged_data.drop(columns=[col for col in columns_to_drop if col in merged_data.columns])
merged_data.dropna(axis=1, how='all', inplace=True)

# Temizlenmiş veriyi kaydet
output_path = "temizlenmis_ve_islenmis_veriler.xlsx"
merged_data.to_excel(output_path, index=False)
print(f"Tüm işlemler tamamlandı, '{output_path}' dosyasına kaydedildi.")
