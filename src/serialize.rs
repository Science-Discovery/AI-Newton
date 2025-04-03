use std::str::FromStr;

use crate::{
    experiments::{DoExpType, ExpStructure, ObjType, Objstructure, Parastructure},
    knowledge::Knowledge,
    language::{AtomExp, Concept, Proposition},
};
use serde::{
    de::MapAccess, ser::SerializeStruct, Deserialize, Deserializer, Serialize, Serializer,
};

impl Serialize for DoExpType {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        (&self.name).serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for DoExpType {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let s = String::deserialize(deserializer)?;
        Self::new_with_name(s).map_err(|e| serde::de::Error::custom(e))
    }
}

impl Serialize for AtomExp {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let s = format!("{}", self);
        serializer.serialize_str(&s)
    }
}

impl<'de> Deserialize<'de> for AtomExp {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let s = String::deserialize(deserializer)?;
        s.parse().map_err(|e| serde::de::Error::custom(e))
    }
}

impl Serialize for Concept {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let s = format!("{}", self);
        serializer.serialize_str(&s)
    }
}

impl<'de> Deserialize<'de> for Concept {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let s = String::deserialize(deserializer)?;
        s.parse().map_err(|e| serde::de::Error::custom(e))
    }
}

impl Serialize for Proposition {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let s = format!("{}", self);
        serializer.serialize_str(&s)
    }
}

impl<'de> Deserialize<'de> for Proposition {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let s = String::deserialize(deserializer)?;
        s.parse().map_err(|e| serde::de::Error::custom(e))
    }
}

impl Serialize for ObjType {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let s = format!("{}", self);
        serializer.serialize_str(&s)
    }
}

impl<'de> Deserialize<'de> for ObjType {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let s = String::deserialize(deserializer)?;
        s.parse().map_err(|e| serde::de::Error::custom(e))
    }
}

impl Serialize for ExpStructure {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        // serialize a ExpStructure, except for the datastructofdata.
        let mut state = serializer.serialize_struct("ExpStructure", 2)?;
        state.serialize_field("expconfig", self.get_ref_expconfig())?;
        state.serialize_field("do_experiment", &self.do_experiment)?;
        state.end()
    }
}

impl<'de> Deserialize<'de> for ExpStructure {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct ExpStructureVisitor;
        impl<'de> serde::de::Visitor<'de> for ExpStructureVisitor {
            type Value = ExpStructure;

            fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
                formatter.write_str("struct ExpStructure")
            }

            fn visit_map<A>(self, mut map: A) -> Result<Self::Value, A::Error>
            where
                A: MapAccess<'de>,
            {
                let mut expconfig = None;
                let mut do_experiment = None;
                while let Some(key) = map.next_key()? {
                    match key {
                        "expconfig" => {
                            if expconfig.is_some() {
                                return Err(serde::de::Error::duplicate_field("expconfig"));
                            }
                            expconfig = Some(map.next_value()?);
                        }
                        "do_experiment" => {
                            if do_experiment.is_some() {
                                return Err(serde::de::Error::duplicate_field("do_experiment"));
                            }
                            do_experiment = Some(map.next_value()?);
                        }
                        _ => {
                            return Err(serde::de::Error::unknown_field(
                                key,
                                &["expconfig", "do_experiment"],
                            ));
                        }
                    }
                }
                let expconfig =
                    expconfig.ok_or_else(|| serde::de::Error::missing_field("expconfig"))?;
                let do_experiment = do_experiment
                    .ok_or_else(|| serde::de::Error::missing_field("do_experiment"))?;
                Ok(ExpStructure::new(expconfig, do_experiment))
            }
        }
        deserializer.deserialize_struct(
            "ExpStructure",
            &["expconfig", "do_experiment"],
            ExpStructureVisitor,
        )
    }
}

impl Serialize for Parastructure {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let s = format!("{}", self);
        serializer.serialize_str(&s)
    }
}

impl<'de> Deserialize<'de> for Parastructure {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let s = String::deserialize(deserializer)?;
        Parastructure::from_str(&s).map_err(|e| serde::de::Error::custom(e))
    }
}

impl Serialize for Objstructure {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let s = format!("{}", self);
        serializer.serialize_str(&s)
    }
}

impl<'de> Deserialize<'de> for Objstructure {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let s = String::deserialize(deserializer)?;
        Objstructure::from_str(&s).map_err(|e| serde::de::Error::custom(e))
    }
}

impl Serialize for Knowledge {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let s = format!("{}", self);
        serializer.serialize_str(&s)
    }
}

impl<'de> Deserialize<'de> for Knowledge {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let s = String::deserialize(deserializer)?;
        s.parse().map_err(|e| serde::de::Error::custom(e))
    }
}
