import React from 'react';
import { motion } from 'framer-motion';

interface Props {
    emotion: string;
}

const EmotionDisplay: React.FC<Props> = ({ emotion }) => {
    return (
        <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="emotion-display"
        >
            <h2>Current Emotion:</h2>
            <motion.div
                className={`emotion-badge ${emotion}`}
                animate={{
                    scale: [1, 1.1, 1],
                    transition: { duration: 2, repeat: Infinity }
                }}
            >
                {emotion}
            </motion.div>
        </motion.div>
    );
};

export default EmotionDisplay;