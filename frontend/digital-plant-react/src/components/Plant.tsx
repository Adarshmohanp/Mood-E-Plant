import React from 'react';
import { motion } from 'framer-motion';

interface Props {
    plantImage: string;
}

const Plant: React.FC<Props> = ({ plantImage }) => {
    return (
        <motion.div
            className="plant-container"
            animate={{ y: [0, -5, 0] }}
            transition={{ duration: 2, repeat: Infinity }}
        >
            <img 
                src={`data:image/png;base64,${plantImage}`} 
                alt="Mood Plant" 
                className="plant-image"
            />
        </motion.div>
    );
};

export default Plant;