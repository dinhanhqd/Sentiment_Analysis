1. Dataset
-	Bộ dữ liệu Twitter_samples của thư viện NLTK :
Bộ dữ liệu Twitter_samples của thư viện NLTK là một tập dữ liệu gồm các tweet được thu thập từ Twitter. Bộ dữ liệu này chứa hơn 1 triệu tweet, được phân loại theo cảm xúc tích cực, tiêu cực, hoặc trung lập.
Bộ dữ liệu Twitter_samples được sử dụng để đào tạo và đánh giá các mô hình phân tích cảm xúc trong văn bản. Bộ dữ liệu này có nhiều ưu điểm, chẳng hạn như:
Kích thước lớn: Bộ dữ liệu này chứa hơn 1 triệu tweet, cung cấp một lượng dữ liệu lớn để đào tạo các mô hình học máy.
Diversity: Bộ dữ liệu này bao gồm các tweet từ nhiều nguồn khác nhau, đại diện cho nhiều loại cảm xúc.
Availability: Bộ dữ liệu này có sẵn miễn phí và có thể dễ dàng tải xuống.
Dưới đây là một số thông tin chi tiết về bộ dữ liệu Twitter_samples:
Kích thước: 1,085,947 tweet
Ngôn ngữ: Tiếng Anh
Cảm xúc: Tích cực, tiêu cực, trung lập
Thư mục: nltk_data/twitter_samples
Link data: https://realpython.com/python-nltk-sentiment-analysis/
-	Dataset tự tạo:
Với data chúng em thu thập từ các review tiếng việt về các quán ăn ở trên internet. Với mục đích chính là thử xây dựng một model ngôn ngữ riêng cho tiếng việt và biết thêm kiếm thức về thu thập dữ liệu và đánh nhãn cho dữ dạng văn bản.
Linkdata:
https://drive.google.com/drive/folders/1oUy6AEY7UgBAjueP3y-_6YKmgJoH26xJ?usp=sharing
-	Dataset từ một bài báo nghiên cứu khoa học về lĩnh vực phân tích tình cảm
Ở bài báo này nhóm tác giả đã sử dụng mô hình Bert để phân tích và cho kết quả rất tốt. Chúng em lấy dataset này về với mục đích là sử dụng dataset chuẩn và tự xây dựng riêng một model khác để tăng thêm kiến thức về môn học này thay vì sử dụng các model đã xây dựng sẵn 
Bộ dữ liệu thu tập từ các dòng tweet về COVID-19 trên nên tảng mạng xã hội Twitter. Dùng API Scrapper Twitter được sử dụng để trích xuất dữ liệu có hashtag (#Covid2019 HOẶC#Covid19 HOẶC corona&virus) từ ngày 20 tháng 1 năm 2020 đến ngày 25 tháng 4 năm 2020
2020
Link data: 
GitHub - savan77/EmotionDetectionBERT: Multi Emotion Detection from COVID-19 Text using BERT
Thông tin về model
![image](https://github.com/user-attachments/assets/f5a78a7b-90fc-403e-aa9b-2b13f3c5bdd6)


Hình 26: Thông tin model




Kết quả
 ![image](https://github.com/user-attachments/assets/7017e83d-b1fa-4e77-82d8-6faf5b2d6a71)

Hình 27: Kết quả
 ![image](https://github.com/user-attachments/assets/158a21f8-b9dc-4975-bc31-c6ec2501d9dc)

Kết qua cho ra rất tốt có độ chính xác lên tới hơn 92%. Có thể thấy model mà chúng em thiết kế khá là phù hợp với bộ dataset này, tuy còn thấp hơn so với mô hình Bert mà nhóm nghiên cứu trong bài báo là 98%. Nhưng bù lại là mô hình của chúng em nhỏ gọn là training nhanh hơn, có thể xem đây là một kết quả khá tốt. 
Đồ thị:
 
Hình 28: Đồ thị
![image](https://github.com/user-attachments/assets/871a81a0-3785-4e6a-8399-6908fcecf0b1)



Web:
![image](https://github.com/user-attachments/assets/c4f59274-72f1-4bb3-ba08-08b4cea68e4b)
![image](https://github.com/user-attachments/assets/9a578390-157f-4eb8-8ed0-682a2dc4b52d)
![image](https://github.com/user-attachments/assets/e941a261-199e-4652-adc4-b9c7bbf80f51)
![image](https://github.com/user-attachments/assets/44ad586d-853d-4533-b19b-6190554bc7e7)
![image](https://github.com/user-attachments/assets/65510271-76d3-4b69-a479-384bf606fc48)
