export interface EmotionResponse {
    emotion: string;
    plant_image: string;
}

export const fetchEmotion = async (): Promise<EmotionResponse> => {
    const response = await fetch('http://localhost:8000/emotion');
    if (!response.ok) {
        throw new Error('Failed to fetch emotion data');
    }
    return response.json();
};