#include <gtest/gtest.h>
#include "ufra/face_parser.h"
#include <opencv2/opencv.hpp>

class FaceParserTest : public ::testing::Test {
protected:
    void SetUp() override {
        parser = std::make_unique<ufra::FaceParser>();
        
        // Create test face crop
        face_crop = cv::Mat::zeros(256, 256, CV_8UC3);
        
        // Create synthetic face-like structure
        cv::ellipse(face_crop, cv::Point(128, 128), cv::Size(80, 100), 0, 0, 360, cv::Scalar(200, 180, 160), -1); // Face
        cv::ellipse(face_crop, cv::Point(100, 110), cv::Size(12, 8), 0, 0, 360, cv::Scalar(50, 50, 50), -1);       // Left eye
        cv::ellipse(face_crop, cv::Point(156, 110), cv::Size(12, 8), 0, 0, 360, cv::Scalar(50, 50, 50), -1);       // Right eye
        cv::ellipse(face_crop, cv::Point(128, 150), cv::Size(20, 15), 0, 0, 180, cv::Scalar(100, 50, 50), -1);     // Mouth
        cv::ellipse(face_crop, cv::Point(128, 90), cv::Size(60, 40), 0, 0, 360, cv::Scalar(100, 80, 60), -1);      // Hair
        
        // Create mock parsing mask for testing region extraction
        mock_parsing_mask = cv::Mat::zeros(256, 256, CV_8UC1);
        cv::circle(mock_parsing_mask, cv::Point(128, 128), 80, 1, -1);   // Skin = 1
        cv::circle(mock_parsing_mask, cv::Point(100, 110), 12, 4, -1);   // Left eye = 4
        cv::circle(mock_parsing_mask, cv::Point(156, 110), 12, 5, -1);   // Right eye = 5
        cv::ellipse(mock_parsing_mask, cv::Point(128, 150), cv::Size(20, 15), 0, 0, 180, 10, -1); // Mouth = 10
        cv::circle(mock_parsing_mask, cv::Point(128, 90), 40, 13, -1);   // Hair = 13
        cv::rectangle(mock_parsing_mask, cv::Point(90, 95), cv::Point(110, 105), 6, -1);  // Left eyebrow = 6
        cv::rectangle(mock_parsing_mask, cv::Point(146, 95), cv::Point(166, 105), 7, -1); // Right eyebrow = 7
    }

    std::unique_ptr<ufra::FaceParser> parser;
    cv::Mat face_crop;
    cv::Mat mock_parsing_mask;
    cv::Mat empty_image;
};

TEST_F(FaceParserTest, CreateParser) {
    ASSERT_NE(parser, nullptr);
}

TEST_F(FaceParserTest, ParseFaceEmptyImage) {
    cv::Mat result = parser->parseFace(empty_image);
    EXPECT_TRUE(result.empty() || result.total() == 0);
}

TEST_F(FaceParserTest, ParseFaceValidImage) {
    cv::Mat result = parser->parseFace(face_crop);
    
    // Without loaded model, should return zeros or handle gracefully
    EXPECT_FALSE(result.empty());
    EXPECT_EQ(result.size(), face_crop.size());
    EXPECT_EQ(result.type(), CV_8UC1);
}

TEST_F(FaceParserTest, ParseFacesBatch) {
    std::vector<cv::Mat> face_crops = {face_crop, face_crop, face_crop};
    std::vector<cv::Mat> results = parser->parseFacesBatch(face_crops);
    
    EXPECT_EQ(results.size(), 3);
    for (const auto& result : results) {
        EXPECT_FALSE(result.empty());
        EXPECT_EQ(result.size(), face_crop.size());
        EXPECT_EQ(result.type(), CV_8UC1);
    }
}

TEST_F(FaceParserTest, GetEyesMask) {
    cv::Mat eyes_mask = parser->getEyesMask(mock_parsing_mask);
    
    EXPECT_FALSE(eyes_mask.empty());
    EXPECT_EQ(eyes_mask.size(), mock_parsing_mask.size());
    EXPECT_EQ(eyes_mask.type(), CV_8UC1);
    
    // Check that eye regions are detected
    EXPECT_GT(cv::sum(eyes_mask)[0], 0); // Should have some white pixels
}

TEST_F(FaceParserTest, GetHairMask) {
    cv::Mat hair_mask = parser->getHairMask(mock_parsing_mask);
    
    EXPECT_FALSE(hair_mask.empty());
    EXPECT_EQ(hair_mask.size(), mock_parsing_mask.size());
    EXPECT_EQ(hair_mask.type(), CV_8UC1);
    
    // Check that hair region is detected
    EXPECT_GT(cv::sum(hair_mask)[0], 0);
}

TEST_F(FaceParserTest, GetMouthMask) {
    cv::Mat mouth_mask = parser->getMouthMask(mock_parsing_mask);
    
    EXPECT_FALSE(mouth_mask.empty());
    EXPECT_EQ(mouth_mask.size(), mock_parsing_mask.size());
    EXPECT_EQ(mouth_mask.type(), CV_8UC1);
}

TEST_F(FaceParserTest, GetEyebrowsMask) {
    cv::Mat eyebrows_mask = parser->getEyebrowsMask(mock_parsing_mask);
    
    EXPECT_FALSE(eyebrows_mask.empty());
    EXPECT_EQ(eyebrows_mask.size(), mock_parsing_mask.size());
    EXPECT_EQ(eyebrows_mask.type(), CV_8UC1);
    
    // Check that eyebrow regions are detected
    EXPECT_GT(cv::sum(eyebrows_mask)[0], 0);
}

TEST_F(FaceParserTest, GetForeheadMask) {
    cv::Mat forehead_mask = parser->getForeheadMask(mock_parsing_mask);
    
    EXPECT_FALSE(forehead_mask.empty());
    EXPECT_EQ(forehead_mask.size(), mock_parsing_mask.size());
    EXPECT_EQ(forehead_mask.type(), CV_8UC1);
}

TEST_F(FaceParserTest, GetCheeksMask) {
    cv::Mat cheeks_mask = parser->getCheeksMask(mock_parsing_mask);
    
    EXPECT_FALSE(cheeks_mask.empty());
    EXPECT_EQ(cheeks_mask.size(), mock_parsing_mask.size());
    EXPECT_EQ(cheeks_mask.type(), CV_8UC1);
}

TEST_F(FaceParserTest, GetJawMask) {
    cv::Mat jaw_mask = parser->getJawMask(mock_parsing_mask);
    
    EXPECT_FALSE(jaw_mask.empty());
    EXPECT_EQ(jaw_mask.size(), mock_parsing_mask.size());
    EXPECT_EQ(jaw_mask.type(), CV_8UC1);
}

TEST_F(FaceParserTest, GetNeckMask) {
    cv::Mat neck_mask = parser->getNeckMask(mock_parsing_mask);
    
    EXPECT_FALSE(neck_mask.empty());
    EXPECT_EQ(neck_mask.size(), mock_parsing_mask.size());
    EXPECT_EQ(neck_mask.type(), CV_8UC1);
}

TEST_F(FaceParserTest, SetInputSize) {
    // Test setting different input sizes
    parser->setInputSize(512, 512);
    parser->setInputSize(256, 256);
    parser->setInputSize(1024, 1024);
    
    // Should not crash and parsing should still work
    cv::Mat result = parser->parseFace(face_crop);
    EXPECT_FALSE(result.empty());
}

TEST_F(FaceParserTest, LoadInvalidModel) {
    bool result = parser->loadModel("nonexistent_model.onnx");
    EXPECT_FALSE(result);
    
    // Parsing should still work (return zeros)
    cv::Mat parsing_result = parser->parseFace(face_crop);
    EXPECT_FALSE(parsing_result.empty());
}

TEST_F(FaceParserTest, RegionMaskConsistency) {
    // Test that all region masks have consistent properties
    std::vector<cv::Mat> region_masks = {
        parser->getEyesMask(mock_parsing_mask),
        parser->getHairMask(mock_parsing_mask),
        parser->getMouthMask(mock_parsing_mask),
        parser->getEyebrowsMask(mock_parsing_mask),
        parser->getForeheadMask(mock_parsing_mask),
        parser->getCheeksMask(mock_parsing_mask),
        parser->getJawMask(mock_parsing_mask),
        parser->getNeckMask(mock_parsing_mask)
    };
    
    for (const auto& mask : region_masks) {
        EXPECT_FALSE(mask.empty());
        EXPECT_EQ(mask.size(), mock_parsing_mask.size());
        EXPECT_EQ(mask.type(), CV_8UC1);
        
        // Check values are binary (0 or 255)
        cv::Mat binary_check;
        cv::inRange(mask, cv::Scalar(0), cv::Scalar(0), binary_check);
        cv::Mat non_binary = mask != 0 & mask != 255;
        EXPECT_EQ(cv::sum(non_binary)[0], 0); // Should be all zeros (all values are 0 or 255)
    }
}