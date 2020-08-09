import React from 'react';

import './Background.css';

interface BackgroundProps {
  children: React.ReactNode;
}

function Background({ children }: BackgroundProps) {
  const [image, setImage] = React.useState(null);

  React.useEffect(() => {
    fetch('https://source.unsplash.com/1920x1080/?Landscape')
      .then((response) => {
        if (!response.ok) {
          throw Error('Error fetching image!');
        }
        return response.url;
      })
      .then((data) => setImage(data as any))
      .catch((err) => {
        throw Error(err.message);
      });
  }, []);

  if (image == null) {
    return <div></div>;
  } else {
    console.log(image);
    return (
      <div className='background-container'>
        <div
          className='background-image'
          style={{ backgroundImage: `url(${image})` }}
        >
          {children}
        </div>
      </div>
    );
  }
}

export { Background };
